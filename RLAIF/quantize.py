from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import wandb
import argparse

def quantize_model():
    from transformers import AutoConfig, LlamaConfig

    # 配置量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    run = wandb.init()
    artifact = run.use_artifact(
        'easonwangzk-the-university-of-chicago/ppo_medical_rl/ppo_final_model:v0',
        type='model'
    )
    artifact_dir = artifact.download()

    # ✅ 强制设定模型类型 + 动态修复vocab_size mismatch问题
    config = LlamaConfig.from_pretrained(artifact_dir)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            artifact_dir,
            config=config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except RuntimeError as e:
        if "size mismatch for weight" in str(e) and "vocab_size" in str(config):
            print("⚠️ Detected vocab size mismatch. Attempting to fix...")
            config.vocab_size = 128258  # match the checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                artifact_dir,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            raise e

    # 同步tokenizer
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, trust_remote_code=True)

    # 保存到本地
    model.save_pretrained("./quantized_model")
    tokenizer.save_pretrained("./quantized_model")

def test_model():
    run = wandb.init(project="ppo_medical_rl", name="quantized_eval")

    from tqdm import tqdm
    from datasets import load_dataset

    # 加载量化后的模型
    model = AutoModelForCausalLM.from_pretrained(
        "./quantized_model",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("./quantized_model", trust_remote_code=True)

    # 加载PubMedQA测试数据
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    split = "train"  # 可根据需要修改为 "validation" 或 "test"
    test_data = dataset[split]

    correct = 0
    total = 0

    for item in tqdm(test_data):
        context = " ".join(item["CONTEXTS"])
        question = item["QUESTION"]
        true_answer = item["final_decision"]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        if "yes" in pred_text:
            pred = "yes"
        elif "no" in pred_text:
            pred = "no"
        elif "maybe" in pred_text:
            pred = "maybe"
        else:
            pred = "unknown"

        if pred == true_answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"PubMedQA Accuracy: {accuracy:.4f}")
    wandb.log({"PubMedQA Accuracy": accuracy})

    import json
    import os

    report = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct
    }
    os.makedirs("eval_artifacts", exist_ok=True)
    with open("eval_artifacts/quantized_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    artifact = wandb.Artifact("quantized_eval_report", type="eval_report")
    artifact.add_file("eval_artifacts/quantized_eval_report.json")
    run.log_artifact(artifact)

    run.finish()

def test_pretrained_model():
    run = wandb.init(project="ppo_medical_rl", name="pretrained_eval")

    from tqdm import tqdm
    from datasets import load_dataset

    artifact = wandb.use_artifact(
        'easonwangzk-the-university-of-chicago/ppo_medical_rl/ppo_final_model:v0',
        type='model'
    )
    artifact_dir = artifact.download()

    from transformers import LlamaConfig
    config = LlamaConfig.from_pretrained(artifact_dir)
    # 动态修复 vocab_size mismatch
    import torch
    import os
    bin_path = os.path.join(artifact_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        checkpoint = torch.load(bin_path, map_location="cpu")
        for key in checkpoint:
            if "embed_tokens.weight" in key:
                config.vocab_size = checkpoint[key].shape[0]
                print(f"🛠️ Patched vocab_size to {config.vocab_size}")
                break
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir,
        config=config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, trust_remote_code=True)

    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    test_data = dataset["train"]

    correct = 0
    total = 0

    for item in tqdm(test_data):
        context = " ".join(item["CONTEXTS"])
        question = item["QUESTION"]
        true_answer = item["final_decision"]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        if "yes" in pred_text:
            pred = "yes"
        elif "no" in pred_text:
            pred = "no"
        elif "maybe" in pred_text:
            pred = "maybe"
        else:
            pred = "unknown"

        if pred == true_answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"PubMedQA Accuracy: {accuracy:.4f}")
    wandb.log({"PubMedQA Accuracy": accuracy})

    import json
    import os

    report = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct
    }
    os.makedirs("eval_artifacts", exist_ok=True)
    with open("eval_artifacts/pretrained_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    artifact = wandb.Artifact("pretrained_eval_report", type="eval_report")
    artifact.add_file("eval_artifacts/pretrained_eval_report.json")
    run.log_artifact(artifact)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and/or test PubMedQA model.")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization and test pretrained model directly.")
    args = parser.parse_args()

    if args.no_quantize:
        test_pretrained_model()
    else:
        quantize_model()
        test_model()