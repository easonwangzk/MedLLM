import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import wandb
from copy import deepcopy
import psutil
import time
from trl.models.modeling_value_head import ValueHead


def get_system_usage():
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)
    if torch.cuda.is_available():
        gpu_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    else:
        gpu_used_gb = 0
        gpu_total_gb = 0
    return ram_used_gb, ram_total_gb, gpu_used_gb, gpu_total_gb

def load_data():
    with open("rlaif_data.json", "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# === Re===
class MedicalRewardFunction:
    def __init__(self):
        self.medical_keywords = {
            'diagnosis': ['diagnose', 'diagnosis', 'condition', 'disease', 'syndrome'],
            'treatment': ['treat', 'treatment', 'therapy', 'medication', 'intervention'],
            'symptoms': ['symptom', 'sign', 'presentation', 'manifestation'],
            'risk_factors': ['risk', 'factor', 'predisposition', 'susceptibility'],
            'prognosis': ['prognosis', 'outcome', 'progression', 'course'],
        }
        self.cot_keywords = [
            'first', 'then', 'next', 'because', 'therefore', 'thus',
            'consequently', 'as a result', 'in conclusion'
        ]

    def calculate_reward(self, text: str) -> float:
        steps = len(re.findall(r'\d+\.|\n-|\n\*', text))
        keywords = sum(1 for word in self.cot_keywords if word in text.lower())
        medical_terms = sum(1 for group in self.medical_keywords.values() for word in group if word in text.lower())
        word_count = len(text.split())

        score = 0.0
        score += min((steps + keywords) / 5, 1.0) * 0.3
        score += min(medical_terms / 10, 1.0) * 0.3
        score += (1.0 if 100 <= word_count <= 300 else 0.5) * 0.4

        return score

# === Main Training Process ===
def train():
    import gc
    dataset = load_data()

    model_name = "skumar9/Llama-medx_v3.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    def preprocess_function(example):
        encoded = tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=512,
            # return_tensors="pt",  # Remove to return lists, not tensors
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    dataset = dataset.map(preprocess_function, batched=False)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    base_model.gradient_checkpointing_enable()
    # Disable cache for base_model
    base_model.config.use_cache = False

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, trust_remote_code=True)
    model.pretrained_model = base_model
    model.v_head = ValueHead(config=base_model.config)
    model.generation_config = base_model.generation_config
    model.base_model_prefix = "pretrained_model"
    model.pretrained_model.gradient_checkpointing_enable()
    # Disable cache for model.pretrained_model
    model.pretrained_model.config.use_cache = False

    # Memory-efficient int4 ref_model
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, trust_remote_code=True, quantization_config=bnb_config
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.base_model_prefix = "pretrained_model"

    # Move ref_model to CPU for memory efficiency
    ref_model.cpu()

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # === Dynamic batch size adjustment based on free GPU memory ===
    import pynvml
    pynvml.nvmlInit()
    free_mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / (1024**3)
    batch_size = 1  # Force smallest batch size to reduce memory pressure

    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=2,
        seed=0,
    )

    import wandb
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(project="ppo_medical_rl", name="ppo_run", config=ppo_config.to_dict(),
               sync_tensorboard=False, settings=wandb.Settings(_disable_stats=True))

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=model,  # ✅ Explicitly pass reward_model
        value_model=model,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Define optimizer (AdamW) after PPOTrainer is initialized
    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=ppo_config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    reward_fn = MedicalRewardFunction()

    scaler = None

    def ppo_rlhf_training_step(model, queries, responses, rewards):
        # Forward
        outputs = model.policy(queries, attention_mask=(queries != tokenizer.pad_token_id).long())
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Take the first element of the tuple
        if isinstance(outputs, torch.Tensor):
            logits = outputs  # If outputs is already a tensor, use it directly
        else:
            logits = outputs.logits

        # 简单计算log_prob
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Align shapes by truncating to the minimum sequence length
        min_len = min(logprobs.shape[1], responses.shape[1])
        aligned_logprobs = logprobs[:, :min_len, :]
        aligned_responses = responses[:, :min_len]

        logprobs_action = torch.gather(aligned_logprobs, dim=2, index=aligned_responses.unsqueeze(-1)).squeeze(-1)
        advantages = rewards - rewards.mean()

        # Policy loss (简单版PPO clip loss，不是真正clip，只近似)
        policy_loss = -(advantages * logprobs_action.mean(dim=1)).mean()

        # Value loss (mock)
        value_loss = torch.tensor(0.0, device=logits.device)

        total_loss = policy_loss + 0.5 * value_loss

        return total_loss

    best_mean_reward = -float('inf')
    early_stopping_patience = 10
    no_improve_steps = 0

    start_time = time.time()
    total_tokens = 0

    device = next(trainer.model.parameters()).device
    for step, batch in enumerate(trainer.get_train_dataloader()):
        queries = batch["input_ids"].to(device, non_blocking=True)

        trainer.model.policy.eval()
        # 生成response
        with torch.no_grad():
            response_tensors = trainer.model.policy.pretrained_model.generate(
                queries,
                max_length=192,
                max_new_tokens=32,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        trainer.model.policy.train()

        total_tokens += response_tensors.shape[1]

        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = [reward_fn.calculate_reward(response) for response in responses]

        mean_reward = sum(rewards) / len(rewards)

        ram_used, ram_total, gpu_used, gpu_total = get_system_usage()

        wandb.log({
            "step": step,
            "mean_reward": mean_reward,
            "ram_used_gb": ram_used,
            "gpu_used_gb": gpu_used
        })
        print(f"[Step {step}] Reward: {mean_reward:.4f} | RAM: {ram_used:.2f}/{ram_total:.2f} GB | GPU: {gpu_used:.2f}/{gpu_total:.2f} GB")

        if step > 0 and step % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(f"[Step {step}] Tokens/sec: {tokens_per_sec:.2f}")

        loss = ppo_rlhf_training_step(trainer.model, queries, response_tensors, torch.tensor(rewards, device=device))
        loss.backward()
        for param in trainer.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None
        queries = queries.cpu()
        response_tensors = response_tensors.cpu()
        torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        optimizer.step()
        torch.cuda.empty_cache()
        gc.collect()
        optimizer.zero_grad()

        wandb.log({
            "step": step,
            "mean_reward": mean_reward,
            "ppo_loss": loss.item(),
            "ram_used_gb": ram_used,
            "gpu_used_gb": gpu_used
        })

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        if no_improve_steps >= early_stopping_patience:
            print(f"Early stopping triggered at step {step}. Best reward: {best_mean_reward:.4f}")
            trainer.save_model(output_dir="./ppo_best_model")
            print("Best model saved to ./ppo_best_model")
            break

        if step > 0 and step % 50 == 0:
            trainer.save_model(output_dir=f"./ppo_checkpoint_step_{step}")
            print(f"Checkpoint saved at step {step}")

    trainer.save_model(output_dir="./ppo_final_model")
    print("Final model saved to ./ppo_final_model")

    artifact = wandb.Artifact('ppo_final_model', type='model')
    artifact.add_dir('./ppo_final_model')
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    train()