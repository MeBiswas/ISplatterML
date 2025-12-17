import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

from config import CATEGORIES
from src.clip_classifier import CLIPClassifier
from src.custom_classifier import CustomClassifier

class HybridClassifier:
    def __init__(self, clip_weight: float = 0.3, custom_weight: float = 0.7):
        """Hybrid classifier combining CLIP and custom model"""
        self.clip_weight = clip_weight
        self.custom_weight = custom_weight
        
        self.clip_model = CLIPClassifier()
        self.custom_model = CustomClassifier()
        
        self.categories = list(CATEGORIES.keys())
        print(f"Hybrid classifier initialized: CLIP({clip_weight}) + Custom({custom_weight})")
    
    def classify_image(self, image_path: str) -> Dict:
        """Combine predictions from both models"""
        # Get predictions from both models
        clip_result = self.clip_model.classify_image(image_path)
        custom_result = self.custom_model.classify_image(image_path)
        
        # Combine scores
        combined_scores = {}
        for category in self.categories:
            clip_score = clip_result["all_scores"].get(category, 0)
            custom_score = custom_result["all_scores"].get(category, 0)
            
            combined_scores[category] = (
                self.clip_weight * clip_score + 
                self.custom_weight * custom_score
            )
        
        # Get best category
        best_category = max(combined_scores.items(), key=lambda x: x[1])
        
        return {
            "category": best_category[0],
            "confidence": float(best_category[1]),
            "all_scores": combined_scores,
            "clip_prediction": clip_result["category"],
            "clip_confidence": clip_result["confidence"],
            "custom_prediction": custom_result["category"],
            "custom_confidence": custom_result["confidence"],
            "model": "Hybrid"
        }