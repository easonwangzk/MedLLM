import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Optional
from configs.model_config import DISTILL_CONFIG

class DistillationTrainer:
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda"
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move models to device
        self.teacher_model.to(device)
        self.student_model.to(device)
        
        # Set teacher model to eval mode
        self.teacher_model.eval()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=5e-5
        )
        
    def compute_distillation_loss(
        self,
        teacher_outputs: Dict,
        student_outputs: Dict
    ) -> torch.Tensor:
        loss = 0.0
        
        # 1. Logits distillation
        teacher_logits = teacher_outputs.logits / DISTILL_CONFIG["temperature"]
        student_logits = student_outputs.logits / DISTILL_CONFIG["temperature"]
        
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        
        logits_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction="batchmean"
        ) * (DISTILL_CONFIG["temperature"] ** 2)
        
        loss += DISTILL_CONFIG["alpha"] * logits_loss
        
        # 2. Attention distillation
        if DISTILL_CONFIG["attention_distill"]:
            for layer_idx in DISTILL_CONFIG["distill_layers"]:
                teacher_attention = teacher_outputs.attentions[layer_idx]
                student_attention = student_outputs.attentions[layer_idx]
                
                attention_loss = F.mse_loss(
                    student_attention,
                    teacher_attention
                )
                loss += DISTILL_CONFIG["alpha"] * attention_loss
        
        # 3. Hidden states distillation
        if DISTILL_CONFIG["hidden_distill"]:
            for layer_idx in DISTILL_CONFIG["distill_layers"]:
                teacher_hidden = teacher_outputs.hidden_states[layer_idx]
                student_hidden = student_outputs.hidden_states[layer_idx]
                
                hidden_loss = F.mse_loss(
                    student_hidden,
                    teacher_hidden
                )
                loss += DISTILL_CONFIG["alpha"] * hidden_loss
        
        return loss
        
    def compute_task_loss(
        self,
        student_outputs: Dict,
        labels: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            labels.view(-1)
        )
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.student_model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
        
        # Get student outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Compute losses
        distill_loss = self.compute_distillation_loss(
            teacher_outputs,
            student_outputs
        )
        task_loss = self.compute_task_loss(student_outputs, labels)
        
        # Total loss
        total_loss = (
            DISTILL_CONFIG["alpha"] * distill_loss +
            DISTILL_CONFIG["beta"] * task_loss
        )
        
        # Update student model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.student_model.parameters(),
            1.0
        )
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "task_loss": task_loss.item()
        }
