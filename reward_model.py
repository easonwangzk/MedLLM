import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import List, Dict
import json

class MedicalRewardModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/BioGPT-Large",
            num_labels=1
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
        
        # Load medical knowledge base
        self.medical_kb = self._load_medical_knowledge_base()
    
    def _load_medical_knowledge_base(self) -> Dict:
        """Load medical knowledge base for fact checking"""
        # This should be replaced with your actual medical knowledge base
        return {}
    
    def evaluate_response(self, response: str, context: str = None) -> float:
        """
        Evaluate a medical response based on multiple criteria
        Returns a score between 0 and 1
        """
        # Tokenize input
        inputs = self.tokenizer(
            response,
            context if context else "",
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            base_score = torch.sigmoid(outputs.logits).item()
        
        # Additional scoring factors
        accuracy_score = self._check_medical_accuracy(response)
        completeness_score = self._check_completeness(response)
        clarity_score = self._check_clarity(response)
        
        # Combine scores with weights
        final_score = (
            0.4 * base_score +
            0.3 * accuracy_score +
            0.2 * completeness_score +
            0.1 * clarity_score
        )
        
        return final_score
    
    def _check_medical_accuracy(self, response: str) -> float:
        """Check if the response contains accurate medical information"""
        # Implement medical fact checking against knowledge base
        return 0.8  # Placeholder
    
    def _check_completeness(self, response: str) -> float:
        """Check if the response covers all necessary aspects"""
        required_elements = [
            "diagnosis",
            "treatment",
            "prognosis",
            "risk factors"
        ]
        score = 0
        for element in required_elements:
            if element in response.lower():
                score += 0.25
        return score
    
    def _check_clarity(self, response: str) -> float:
        """Check if the response is clear and well-structured"""
        # Implement clarity checking logic
        return 0.9  # Placeholder

def get_reward(response: str, context: str = None) -> float:
    """Get reward for a response using the medical reward model"""
    reward_model = MedicalRewardModel()
    return reward_model.evaluate_response(response, context) 