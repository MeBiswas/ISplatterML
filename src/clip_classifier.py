import clip
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from config import CATEGORIES, CLIP_CONFIG, MODEL_CONFIGS

class CLIPClassifier:
    def __init__(self, model_name: str = "clip"):
        self.config = MODEL_CONFIGS[model_name]
        self.device = torch.device(self.config["device"])

        print(f"Loading CLIP model: ViT-B/32 on {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.category_names = list(CATEGORIES.keys())
        self.prompts = self._prepare_prompts()
        
        print(f"CLIP model loaded on {self.device}")

    def _prepare_prompts(self) -> torch.Tensor:
        """Convert all prompts to CLIP text embeddings"""
        all_prompts = []
        for category_name, category_info in CATEGORIES.items():
            # Use multiple prompts per category for robustness
            for prompt in category_info["clip_prompts"][:CLIP_CONFIG["top_k"]]:
                all_prompts.append(prompt)
        
        # Encode all prompts
        text_tokens = clip.tokenize(all_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            if CLIP_CONFIG["normalize"]:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def classify_image(self, image_path: str) -> Dict:
        """Classify a single image using CLIP"""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            if CLIP_CONFIG["normalize"]:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity with all prompts
            similarity = (100.0 * image_features @ self.prompts.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0]
        
        # Aggregate by category (average of prompts per category)
        category_scores = {}
        prompt_idx = 0
        for category_name, category_info in CATEGORIES.items():
            num_prompts = min(CLIP_CONFIG["top_k"], len(category_info["clip_prompts"]))
            cat_score = np.mean(similarity[prompt_idx:prompt_idx + num_prompts])
            category_scores[category_name] = cat_score
            prompt_idx += num_prompts
        
        # Get best category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        return {
            "category": best_category[0],
            "confidence": float(best_category[1]),
            "all_scores": category_scores,
            "model": "CLIP"
        }
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict]:
        """Classify multiple images (more efficient)"""
        # Batch processing implementation
        results = []
        for img_path in image_paths:
            try:
                result = self.classify_image(img_path)
                result["image_path"] = img_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    "image_path": img_path,
                    "category": "error",
                    "confidence": 0.0,
                    "error": str(e)
                })
        return results