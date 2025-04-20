from transformers import PreTrainedTokenizer
from typing import Dict, List
import torch
from datasets import Dataset

class PubMedQADataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = Dataset.load_from_disk(f"data/pubmed_qa")[split]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 构建输入文本
        text = f"Context: {item['context']}\n\nQuestion: {item['question']}\n\nAnswer:"
        
        # 编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 添加标签
        encoding["labels"] = self.tokenizer(
            item["long_answer"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["labels"].squeeze()
        }
