import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
import wandb


# Processor class for DPOTrainer
class DPOProcessor:
    def __init__(self, tokenizer, max_length=512):
        self._tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, example):
        # 注意：这里直接处理单个样本，而不是批量
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # 处理 prompt
        prompt_tokens = self._tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

        # 处理 chosen 和 rejected
        chosen_tokens = self._tokenizer(
            prompt + chosen,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,  # 不要返回 tensor
        )

        rejected_tokens = self._tokenizer(
            prompt + rejected,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,  # 不要返回 tensor
        )

        return {
            "input_ids": prompt_tokens["input_ids"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }

def load_dpo_data():
    with open("rlaif_data.json", "r") as f:
        raw_data = json.load(f)
    formatted_data = []
    
    for item in raw_data:
        try:
            # 确保数据格式正确
            if isinstance(item, dict) and all(k in item for k in ["prompt", "chosen", "rejected"]):
                chosen_text = item["chosen"]["chain_of_thought"] + "\n" + item["chosen"]["answer"]
                rejected_text = item["rejected"]["answer"]
                
                if all(isinstance(x, str) for x in [item["prompt"], chosen_text, rejected_text]):
                    formatted_data.append({
                        "prompt": item["prompt"],
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    })
        except (KeyError, TypeError, AttributeError) as e:
            print(f"Skipping malformed data item: {e}")
            continue
            
    print(f"Loaded {len(formatted_data)} valid training examples")
    return Dataset.from_list(formatted_data)

def train_dpo():
    # 1. 基础设置
    model_name = "skumar9/Llama-medx_v3.2"
    
    # 2. 配置 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 3. 加载和配置模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # 4. 配置参考模型（使用 4bit 量化）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    # 5. DPO 配置
    dpo_config = DPOConfig(
        beta=0.1,
        learning_rate=5e-6,
        max_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_dir="./dpo_logs",
        padding_value=tokenizer.pad_token_id,
        gradient_checkpointing=True
    )

    # 6. 准备数据和处理器
    dataset = load_dpo_data()
    processor = DPOProcessor(tokenizer)

    # 7. 配置训练器
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=processor,
    )

    # 8. 配置 wandb 日志
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(
        project="dpo_medical_rl",
        name="dpo_run",
        config=dpo_config.to_dict()
    )

    # 9. 训练和保存
    trainer.train()
    trainer.save_model(output_dir="./dpo_model")
    
    # 10. 保存 artifact
    artifact = wandb.Artifact('dpo_model', type='model')
    artifact.add_dir('./dpo_model')
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    train_dpo()
