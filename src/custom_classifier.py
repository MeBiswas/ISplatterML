import torch
from PIL import Image
import torch.nn as nn
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as transforms

from typing import Dict, List
from config import CATEGORIES, MODEL_CONFIGS

class CustomClassifier:
    def __init__(self, model_type: str = "custom_resnet"):
        """Initialize custom trained model"""
        self.config = MODEL_CONFIGS[model_type]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = list(CATEGORIES.keys())
        
        # Load model
        self.model = self._load_model()
        self.transform = self._get_transform()
        
        print(f"{self.config['name']} loaded on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load pre-trained custom model"""
        if "resnet" in self.config["base_model"]:
            model = getattr(models, self.config["base_model"])(pretrained=False)
            # Modify for our number of classes
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.config["num_classes"])
        elif "efficientnet" in self.config["base_model"]:
            model = getattr(models, self.config["base_model"])(pretrained=False)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.config["num_classes"])
        else:
            raise ValueError(f"Unknown base model: {self.config['base_model']}")
        
        # Load trained weights
        checkpoint = torch.load(self.config["checkpoint"], map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self):
        """Get image transformations (must match training)"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def classify_image(self, image_path: str) -> Dict:
        """Classify image using custom trained model"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get all probabilities
            all_probs = probabilities[0].cpu().numpy()
            category_scores = {
                cat: float(all_probs[idx]) 
                for cat, idx in [(cat, CATEGORIES[cat]["id"]) 
                               for cat in self.categories]
            }
        
        return {
            "category": self.categories[predicted_idx.item()],
            "confidence": confidence.item(),
            "all_scores": category_scores,
            "model": self.config["name"]
        }