import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Optional
from configs.model_config import RLAIF_CONFIG

class RLAIFTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_model: PreTrainedModel,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.device = device
        
        # Move models to device
        self.model.to(device)
        self.reward_model.to(device)
        
        # Set reward model to eval mode
        self.reward_model.eval()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-5
        )
    
    def compute_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            # Get reward model outputs
            outputs = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=generated_ids
            )
            
            # Compute reward (negative loss)
            reward = -outputs.loss
            
            # Apply threshold
            reward = torch.clamp(
                reward,
                min=RLAIF_CONFIG["reward_threshold"]
            )
            
            return reward
    
    def compute_kl_penalty(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor
    ) -> torch.Tensor:
        # Compute KL divergence
        kl_div = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction="batchmean"
        )
        
        return RLAIF_CONFIG["kl_coef"] * kl_div
    
    def compute_entropy_penalty(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        return -RLAIF_CONFIG["entropy_coef"] * entropy
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Generate responses
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=RLAIF_CONFIG["max_length"],
            num_return_sequences=RLAIF_CONFIG["num_rollouts"],
            do_sample=True,
            temperature=RLAIF_CONFIG["temperature"]
        )
        
        # Compute rewards
        rewards = self.compute_reward(
            input_ids,
            attention_mask,
            outputs
        )
        
        # Get model outputs
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Compute policy loss
        policy_loss = -torch.mean(rewards * model_outputs.loss)
        
        # Compute KL penalty
        with torch.no_grad():
            ref_outputs = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        kl_penalty = self.compute_kl_penalty(
            model_outputs.logits,
            ref_outputs.logits
        )
        
        # Compute entropy penalty
        entropy_penalty = self.compute_entropy_penalty(
            model_outputs.logits
        )
        
        # Total loss
        total_loss = policy_loss + kl_penalty + entropy_penalty
        
        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            1.0
        )
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "entropy": entropy_penalty.item()
        }
