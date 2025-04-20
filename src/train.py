import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from data.data_processor import PubMedQADataset
from training.rlaif_trainer import RLAIFTrainer
from training.distillation_trainer import DistillationTrainer
from configs.model_config import MODEL_CONFIG, TRAINING_CONFIG
import logging
import os
from tqdm import tqdm
import time
import gc

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    start_time = time.time()
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 创建必要的目录
    os.makedirs("offload", exist_ok=True)
    os.makedirs("offload_student", exist_ok=True)
    os.makedirs("offload_reward", exist_ok=True)
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    
    # 设置量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=True,  # 启用CPU卸载
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 设置设备映射
    device_map = {
        'model.embed_tokens': 0,
        'model.norm': 0,
        'lm_head': 0,
    }
    
    # 动态分配层到GPU和CPU
    num_layers = 80  # Llama-70B的层数
    gpu_layers = 20  # GPU上保留的层数
    
    # 前20层放在GPU上，其余放在CPU上
    for i in range(num_layers):
        if i < gpu_layers:
            device_map[f'model.layers.{i}'] = 0
        else:
            device_map[f'model.layers.{i}'] = 'cpu'

    # 设置内存限制
    max_memory = {
        0: "35GB",      # GPU内存限制
        'cpu': "100GB"  # CPU内存限制
    }
    
    # Initialize tokenizer and models
    logger.info("Initializing tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["base_model"],
        trust_remote_code=True
    )
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["teacher_model"],
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map=device_map,
        use_cache=False,
        max_memory=max_memory,
        offload_folder="offload",
        trust_remote_code=True
    )
    teacher_model.gradient_checkpointing_enable()
    
    # Load student model
    logger.info("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["student_model"],
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map=device_map,
        use_cache=False,
        max_memory=max_memory,
        offload_folder="offload_student",
        trust_remote_code=True
    )
    student_model.gradient_checkpointing_enable()
    
    # Load reward model
    logger.info("Loading reward model...")
    reward_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["teacher_model"],
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map=device_map,
        use_cache=False,
        max_memory=max_memory,
        offload_folder="offload_reward",
        trust_remote_code=True
    )
    reward_model.gradient_checkpointing_enable()
    
    # Create datasets
    logger.info("Creating datasets...")
    train_datasets = {
        "pqaa": PubMedQADataset(tokenizer, "pqaa", "train"),
        "pqal": PubMedQADataset(tokenizer, "pqal", "train"),
        "pqau": PubMedQADataset(tokenizer, "pqau", "train")
    }
    
    # Create data loaders
    train_loaders = {
        name: DataLoader(
            dataset,
            batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        for name, dataset in train_datasets.items()
    }
    
    # Initialize trainers
    logger.info("Initializing trainers...")
    distillation_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer
    )
    
    rlaif_trainer = RLAIFTrainer(
        model=student_model,
        tokenizer=tokenizer,
        reward_model=reward_model
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(TRAINING_CONFIG["num_train_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{TRAINING_CONFIG['num_train_epochs']}")
        
        # 1. Distillation training
        logger.info("Starting distillation training...")
        for dataset_name, train_loader in train_loaders.items():
            logger.info(f"Distillation training on {dataset_name}...")
            pbar = tqdm(train_loader, desc=f"Distillation {dataset_name}")
            
            for batch in pbar:
                if time.time() - start_time > 3600:  # 1小时限制
                    logger.info("Time limit reached for distillation. Moving to RLAIF...")
                    break
                
                metrics = distillation_trainer.train_step(batch)
                pbar.set_postfix({
                    "total_loss": metrics["total_loss"],
                    "distill_loss": metrics["distill_loss"],
                    "task_loss": metrics["task_loss"]
                })
                
                if pbar.n % TRAINING_CONFIG["logging_steps"] == 0:
                    logger.info(
                        f"Step {pbar.n}: "
                        f"Total Loss: {metrics['total_loss']:.4f}, "
                        f"Distill Loss: {metrics['distill_loss']:.4f}, "
                        f"Task Loss: {metrics['task_loss']:.4f}"
                    )
        
        # 2. RLAIF training
        logger.info("Starting RLAIF training...")
        for dataset_name, train_loader in train_loaders.items():
            logger.info(f"RLAIF training on {dataset_name}...")
            pbar = tqdm(train_loader, desc=f"RLAIF {dataset_name}")
            
            for batch in pbar:
                if time.time() - start_time > 7200:  # 2小时总限制
                    logger.info("Time limit reached. Stopping training...")
                    return
                
                metrics = rlaif_trainer.train_step(batch)
                pbar.set_postfix({
                    "policy_loss": metrics["policy_loss"],
                    "entropy": metrics["entropy"]
                })
                
                if pbar.n % TRAINING_CONFIG["logging_steps"] == 0:
                    logger.info(
                        f"Step {pbar.n}: "
                        f"Policy Loss: {metrics['policy_loss']:.4f}, "
                        f"Entropy: {metrics['entropy']:.4f}"
                    )
                
                if pbar.n % TRAINING_CONFIG["save_steps"] == 0:
                    save_path = os.path.join(
                        TRAINING_CONFIG["output_dir"],
                        f"checkpoint-{epoch}-{dataset_name}-{pbar.n}"
                    )
                    student_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                    
                    torch.cuda.empty_cache()
                    gc.collect()
    
    # Save final model
    final_save_path = os.path.join(TRAINING_CONFIG["output_dir"], "final_model")
    student_model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Saved final model to {final_save_path}")

if __name__ == "__main__":
    main()