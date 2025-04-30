import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from src.data.data_processor import MedicalQADataset
from src.training.distillation_trainer import DistillationTrainer
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.model_config import MODEL_CONFIG, TRAINING_CONFIG

import logging
import os
from tqdm import tqdm
import time
import gc
import wandb

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    # 初始化wandb
    wandb.init(project="medical-llm-distillation")
    
    logger = setup_logging()
    start_time = time.time()
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 初始化tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["teacher_model"],
        trust_remote_code=True
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载教师模型
    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["teacher_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # 使用eager attention
        use_cache=False  # 禁用缓存以兼容gradient checkpointing
    )
    teacher_model.gradient_checkpointing_enable()
    
    # 加载学生模型
    logger.info("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["student_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # 使用eager attention
        use_cache=False  # 禁用缓存以兼容gradient checkpointing
    )
    student_model.gradient_checkpointing_enable()
    
    # 创建数据集
    logger.info("Creating datasets...")
    train_dataset = MedicalQADataset(
        tokenizer=tokenizer,
        split="train",
        max_length=MODEL_CONFIG["max_length"]
    )
    
    val_dataset = MedicalQADataset(
        tokenizer=tokenizer,
        split="validation",
        max_length=MODEL_CONFIG["max_length"]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True  # 丢弃不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True  # 丢弃不完整的批次
    )
    
    # 初始化训练器
    logger.info("Initializing trainer...")
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer
    )
    
    # 训练循环
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(TRAINING_CONFIG["num_train_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{TRAINING_CONFIG['num_train_epochs']}")
        
        # 训练阶段
        trainer.student_model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Ensure batch tensors are on the correct device and have proper dimensions
                processed_batch = {}
                for k, v in batch.items():
                    # Move to device
                    v = v.to(student_model.device)
                    
                    # Ensure proper dimensions
                    if v.dim() == 1:
                        v = v.unsqueeze(0)
                    elif v.dim() == 2:
                        # Add batch dimension if missing
                        v = v.unsqueeze(0)
                    elif v.dim() == 3:
                        # Ensure consistent dimensions with other tensors
                        if k == "input_ids":
                            # For input_ids, we want to keep the sequence dimension
                            pass
                        else:
                            # For other tensors, we might need to adjust
                            v = v.squeeze(0)
                    elif v.dim() == 5:
                        # Handle 5D tensors by squeezing the extra dimension
                        v = v.squeeze(2)  # Remove the middle dimension
                    
                    # Verify tensor size
                    if v.size(-1) != MODEL_CONFIG["max_length"]:
                        # Pad or truncate to max_length
                        if v.size(-1) < MODEL_CONFIG["max_length"]:
                            if v.dim() == 2:
                                padding = torch.zeros(v.size(0), MODEL_CONFIG["max_length"] - v.size(-1), 
                                                    dtype=v.dtype, device=v.device)
                            else:  # dim == 3 or 4
                                padding = torch.zeros(v.size(0), v.size(1), MODEL_CONFIG["max_length"] - v.size(-1), 
                                                    dtype=v.dtype, device=v.device)
                            v = torch.cat([v, padding], dim=-1)
                        else:
                            v = v[..., :MODEL_CONFIG["max_length"]]
                    
                    # Ensure consistent batch size
                    if k == "input_ids" and v.size(0) != TRAINING_CONFIG["per_device_train_batch_size"]:
                        # Pad or truncate batch size
                        if v.size(0) < TRAINING_CONFIG["per_device_train_batch_size"]:
                            if v.dim() == 2:
                                padding = torch.zeros(TRAINING_CONFIG["per_device_train_batch_size"] - v.size(0), 
                                                    v.size(1), dtype=v.dtype, device=v.device)
                            else:  # dim == 3 or 4
                                padding = torch.zeros(TRAINING_CONFIG["per_device_train_batch_size"] - v.size(0), 
                                                    v.size(1), v.size(2), dtype=v.dtype, device=v.device)
                            v = torch.cat([v, padding], dim=0)
                        else:
                            v = v[:TRAINING_CONFIG["per_device_train_batch_size"]]
                    
                    processed_batch[k] = v
                
                # Verify batch consistency
                batch_size = processed_batch["input_ids"].size(0)
                for k, v in processed_batch.items():
                    if v.size(0) != batch_size:
                        # Adjust tensor size to match batch size
                        if v.size(0) < batch_size:
                            if v.dim() == 2:
                                padding = torch.zeros(batch_size - v.size(0), v.size(1), 
                                                    dtype=v.dtype, device=v.device)
                            else:  # dim == 3 or 4
                                padding = torch.zeros(batch_size - v.size(0), v.size(1), v.size(2), 
                                                    dtype=v.dtype, device=v.device)
                            v = torch.cat([v, padding], dim=0)
                        else:
                            v = v[:batch_size]
                        processed_batch[k] = v
                
                # Ensure all tensors have the same number of dimensions
                ref_dim = processed_batch["input_ids"].dim()
                for k, v in processed_batch.items():
                    if v.dim() != ref_dim:
                        if v.dim() < ref_dim:
                            # Add missing dimensions
                            for _ in range(ref_dim - v.dim()):
                                v = v.unsqueeze(-1)
                        else:
                            # Remove extra dimensions
                            while v.dim() > ref_dim:
                                v = v.squeeze(-1)
                        processed_batch[k] = v
                
                metrics = trainer.train_step(processed_batch)
                train_losses.append(metrics["total_loss"])
                
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "distill_loss": f"{metrics['distill_loss']:.4f}",
                    "task_loss": f"{metrics['task_loss']:.4f}"
                })
                
                # 记录到wandb
                wandb.log({
                    "train_total_loss": metrics["total_loss"],
                    "train_distill_loss": metrics["distill_loss"],
                    "train_task_loss": metrics["task_loss"]
                })
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue
        
        if not train_losses:
            logger.warning("No valid training batches processed in this epoch")
            continue
            
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # 验证阶段
        trainer.student_model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    # Ensure batch tensors are on the correct device and have proper dimensions
                    processed_batch = {}
                    for k, v in batch.items():
                        # Move to device
                        v = v.to(student_model.device)
                        
                        # Ensure proper dimensions
                        if v.dim() == 1:
                            v = v.unsqueeze(0)
                        elif v.dim() == 2:
                            # Add batch dimension if missing
                            v = v.unsqueeze(0)
                        elif v.dim() == 3:
                            # Ensure consistent dimensions with other tensors
                            if k == "input_ids":
                                # For input_ids, we want to keep the sequence dimension
                                pass
                            else:
                                # For other tensors, we might need to adjust
                                v = v.squeeze(0)
                        elif v.dim() == 5:
                            # Handle 5D tensors by squeezing the extra dimension
                            v = v.squeeze(2)  # Remove the middle dimension
                        
                        # Verify tensor size
                        if v.size(-1) != MODEL_CONFIG["max_length"]:
                            # Pad or truncate to max_length
                            if v.size(-1) < MODEL_CONFIG["max_length"]:
                                if v.dim() == 2:
                                    padding = torch.zeros(v.size(0), MODEL_CONFIG["max_length"] - v.size(-1), 
                                                        dtype=v.dtype, device=v.device)
                                else:  # dim == 3 or 4
                                    padding = torch.zeros(v.size(0), v.size(1), MODEL_CONFIG["max_length"] - v.size(-1), 
                                                        dtype=v.dtype, device=v.device)
                                v = torch.cat([v, padding], dim=-1)
                            else:
                                v = v[..., :MODEL_CONFIG["max_length"]]
                        
                        # Ensure consistent batch size
                        if k == "input_ids" and v.size(0) != TRAINING_CONFIG["per_device_eval_batch_size"]:
                            # Pad or truncate batch size
                            if v.size(0) < TRAINING_CONFIG["per_device_eval_batch_size"]:
                                if v.dim() == 2:
                                    padding = torch.zeros(TRAINING_CONFIG["per_device_eval_batch_size"] - v.size(0), 
                                                        v.size(1), dtype=v.dtype, device=v.device)
                                else:  # dim == 3 or 4
                                    padding = torch.zeros(TRAINING_CONFIG["per_device_eval_batch_size"] - v.size(0), 
                                                        v.size(1), v.size(2), dtype=v.dtype, device=v.device)
                                v = torch.cat([v, padding], dim=0)
                            else:
                                v = v[:TRAINING_CONFIG["per_device_eval_batch_size"]]
                        
                        processed_batch[k] = v
                    
                    # Verify batch consistency
                    batch_size = processed_batch["input_ids"].size(0)
                    for k, v in processed_batch.items():
                        if v.size(0) != batch_size:
                            # Adjust tensor size to match batch size
                            if v.size(0) < batch_size:
                                if v.dim() == 2:
                                    padding = torch.zeros(batch_size - v.size(0), v.size(1), 
                                                        dtype=v.dtype, device=v.device)
                                else:  # dim == 3 or 4
                                    padding = torch.zeros(batch_size - v.size(0), v.size(1), v.size(2), 
                                                        dtype=v.dtype, device=v.device)
                                v = torch.cat([v, padding], dim=0)
                            else:
                                v = v[:batch_size]
                            processed_batch[k] = v
                    
                    # Ensure all tensors have the same number of dimensions
                    ref_dim = processed_batch["input_ids"].dim()
                    for k, v in processed_batch.items():
                        if v.dim() != ref_dim:
                            if v.dim() < ref_dim:
                                # Add missing dimensions
                                for _ in range(ref_dim - v.dim()):
                                    v = v.unsqueeze(-1)
                            else:
                                # Remove extra dimensions
                                while v.dim() > ref_dim:
                                    v = v.squeeze(-1)
                            processed_batch[k] = v
                    
                    metrics = trainer.train_step(processed_batch)  # 使用train_step但不更新参数
                    val_losses.append(metrics["total_loss"])
                    
                    # 记录到wandb
                    wandb.log({
                        "val_total_loss": metrics["total_loss"],
                        "val_distill_loss": metrics["distill_loss"],
                        "val_task_loss": metrics["task_loss"]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing validation batch {batch_idx}: {str(e)}")
                    continue
        
        if not val_losses:
            logger.warning("No valid validation batches processed in this epoch")
            continue
            
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        logger.info(
            f"Epoch {epoch + 1} - "
            f"Train loss: {avg_train_loss:.4f}, "
            f"Val loss: {avg_val_loss:.4f}"
        )
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(TRAINING_CONFIG["output_dir"], "best_model")
            trainer.save_model(save_path)
            logger.info(f"Saved best model to {save_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(
                TRAINING_CONFIG["output_dir"],
                f"checkpoint-epoch-{epoch + 1}"
            )
            trainer.save_model(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
        
        # 检查是否达到时间限制
        if time.time() - start_time > 7200:  # 2小时限制
            logger.info("Time limit reached. Stopping training...")
            break
    
    # 保存最终模型
    final_save_path = os.path.join(TRAINING_CONFIG["output_dir"], "final_model")
    trainer.save_model(final_save_path)
    logger.info(f"Saved final model to {final_save_path}")
    
    # 关闭wandb
    wandb.finish()

if __name__ == "__main__":
    main() 