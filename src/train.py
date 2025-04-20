import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from transformers import BitsAndBytesConfig
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
    
    # Initialize tokenizer and models
    logger.info("Initializing tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["base_model"])
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["teacher_model"],
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        use_cache=False
    )
    teacher_model.gradient_checkpointing_enable()
    
    # Load student model
    logger.info("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["student_model"],
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        use_cache=False
    )
    student_model.gradient_checkpointing_enable()
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = PubMedQADataset(tokenizer, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
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
        reward_model=teacher_model
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(TRAINING_CONFIG["num_train_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{TRAINING_CONFIG['num_train_epochs']}")
        
        # 1. Distillation training
        logger.info("Starting distillation training...")
        pbar = tqdm(train_loader, desc="Distillation")
        
        for batch in pbar:
            if time.time() - start_time > 3600:  # 1 hour limit
                logger.info("Time limit reached for distillation. Moving to RLAIF...")
                break
            
            metrics = distillation_trainer.train_step(batch)
            pbar.set_postfix({
                "total_loss": metrics["total_loss"],
                "distill_loss": metrics["distill_loss"],
                "task_loss": metrics["task_loss"]
            })
        
        # 2. RLAIF training
        logger.info("Starting RLAIF training...")
        pbar = tqdm(train_loader, desc="RLAIF")
        
        for batch in pbar:
            if time.time() - start_time > 7200:  # 2 hour total limit
                logger.info("Time limit reached. Stopping training...")
                return
            
            metrics = rlaif_trainer.train_step(batch)
            pbar.set_postfix({
                "policy_loss": metrics["policy_loss"],
                "entropy": metrics["entropy"]
            })
            
            if pbar.n % TRAINING_CONFIG["save_steps"] == 0:
                save_path = os.path.join(
                    TRAINING_CONFIG["output_dir"],
                    f"checkpoint-{epoch}-{pbar.n}"
                )
                student_model.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")
                
                torch.cuda.empty_cache()
                gc.collect()
    
    # Save final model
    final_save_path = os.path.join(TRAINING_CONFIG["output_dir"], "final_model")
    student_model.save_pretrained(final_save_path)
    logger.info(f"Saved final model to {final_save_path}")

if __name__ == "__main__":
    main()
