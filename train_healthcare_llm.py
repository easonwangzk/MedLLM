import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import wandb
import numpy as np
from tqdm import tqdm
import os

# Initialize wandb
wandb.init(project="healthcare-llm")

def load_and_preprocess_data():
    """Load and preprocess medical QA datasets"""
    # Load PubMedQA
    pubmed_qa = load_dataset("pubmed_qa", "pqa_labeled")
    
    # Load MedQA
    med_qa = load_dataset("med_qa")
    
    # Load MedMCQA
    med_mcqa = load_dataset("medmcqa")
    
    # Combine datasets
    combined_dataset = concatenate_datasets([
        pubmed_qa["train"],
        med_qa["train"],
        med_mcqa["train"]
    ])
    
    return combined_dataset

def create_prompt_template(example):
    """Create standardized prompt for medical QA"""
    return f"""You are a medical expert. Please provide a detailed and accurate response to the following medical question:

Question: {example['question']}

Context: {example.get('context', '')}

Answer:"""

def train_model():
    # Load base model
    model_name = "microsoft/BioGPT-Large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load and preprocess data
    dataset = load_and_preprocess_data()
    
    # Configure PPO
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        kl_penalty="kl",
        seed=42,
        use_score_scaling=True,
        use_score_norm=True,
        score_clip=None,
    )
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Training loop
    for epoch in range(3):  # Adjust number of epochs as needed
        for batch in tqdm(ppo_trainer.dataloader):
            # Get responses from model
            query_tensors = [tokenizer.encode(create_prompt_template(example), return_tensors="pt") for example in batch]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                length_sampler=lambda x: 100,  # Adjust max length as needed
            )
            
            # Get rewards from reward model (implement your reward model here)
            rewards = [get_reward(response) for response in response_tensors]
            
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log metrics
            wandb.log(stats)
    
    # Save the model
    model.save_pretrained("healthcare_llm")
    tokenizer.save_pretrained("healthcare_llm")

def get_reward(response):
    """Implement your reward model here"""
    # This should be replaced with your actual reward model
    # For now, returning a placeholder reward
    return 0.5

if __name__ == "__main__":
    train_model() 