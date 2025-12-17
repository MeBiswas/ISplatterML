import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from config import CATEGORIES
from typing import Dict, List, Tuple
from src.clip_classifier import CLIPClassifier
from src.custom_classifier import CustomClassifier
from src.hybrid_classifier import HybridClassifier
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, test_data_dir: str = None, ground_truth_dir: str = None):
        """
        Evaluate and compare different models
        
        Args:
            test_data_dir: Directory with test images
            ground_truth_dir: Directory with ground truth labels (same structure as training)
        """
        self.test_dir = Path(test_data_dir) if test_data_dir else None
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None
        self.categories = list(CATEGORIES.keys())
        
        # Initialize models
        self.models = {
            "CLIP": CLIPClassifier(),
            "Custom_ResNet": CustomClassifier("custom_resnet"),
            "Custom_EfficientNet": CustomClassifier("custom_efficientnet"),
            "Hybrid": HybridClassifier()
        }
    
    def evaluate_model(self, model_name: str, model_instance, test_images: List[Path]) -> Dict:
        """Evaluate a single model on test images"""
        print(f"Evaluating {model_name}...")
        
        predictions = []
        ground_truths = []
        confidences = []
        inference_times = []
        
        for img_path in test_images:
            # Get ground truth from folder structure
            if self.ground_truth_dir:
                gt_category = self._get_ground_truth(img_path)
                if gt_category is None:
                    continue
                ground_truths.append(gt_category)
            
            # Get prediction
            import time
            start_time = time.time()
            result = model_instance.classify_image(str(img_path))
            inference_times.append(time.time() - start_time)
            
            predictions.append(result["category"])
            confidences.append(result["confidence"])
        
        # Calculate metrics
        accuracy = None
        if ground_truths:
            accuracy = np.mean([p == gt for p, gt in zip(predictions, ground_truths)])
        
        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "avg_inference_time": np.mean(inference_times) if inference_times else 0,
            "total_images": len(test_images),
            "predictions": predictions,
            "ground_truths": ground_truths if ground_truths else None,
            "confidences": confidences
        }
    
    def _get_ground_truth(self, image_path: Path) -> str:
        """Extract ground truth from folder structure"""
        if not self.ground_truth_dir:
            return None
        
        # Find which category folder contains this image
        for category in self.categories:
            category_dir = self.ground_truth_dir / category
            if category_dir.exists() and image_path.name in [f.name for f in category_dir.iterdir()]:
                return category
        
        # Check if image is in a subdirectory that matches a category
        for parent in image_path.parents:
            if parent.name in self.categories:
                return parent.name
        
        return None
    
    def compare_all_models(self, num_images: int = 100) -> Dict:
        """Compare all models on the same set of images"""
        # Get test images
        if self.test_dir:
            test_images = self._get_test_images(self.test_dir, num_images)
        elif self.ground_truth_dir:
            test_images = self._get_test_images(self.ground_truth_dir, num_images)
        else:
            raise ValueError("No test data directory provided")
        
        print(f"Testing on {len(test_images)} images...")
        
        results = {}
        for model_name, model in self.models.items():
            try:
                results[model_name] = self.evaluate_model(model_name, model, test_images)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {
                    "model_name": model_name,
                    "error": str(e)
                }
        
        # Find best model
        if any('accuracy' in r and r['accuracy'] is not None for r in results.values()):
            valid_results = {k: v for k, v in results.items() 
                           if 'accuracy' in v and v['accuracy'] is not None}
            if valid_results:
                best_model = max(valid_results.items(), 
                               key=lambda x: x[1]['accuracy'])
                results['best_model'] = best_model[1]
        
        return results
    
    def _get_test_images(self, directory: Path, max_images: int) -> List[Path]:
        """Get test images from directory"""
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        all_images = []
        
        for ext in extensions:
            all_images.extend(directory.rglob(f"*{ext}"))
            all_images.extend(directory.rglob(f"*{ext.upper()}"))
        
        # Limit number of images
        if len(all_images) > max_images:
            import random
            all_images = random.sample(all_images, max_images)
        
        return all_images
    
    def generate_report(self, results: Dict, output_dir: str = "reports"):
        """Generate evaluation report with visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = output_path / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects
            serializable_results = {}
            for model_name, result in results.items():
                if model_name == 'best_model':
                    continue
                serializable_results[model_name] = {
                    k: v for k, v in result.items() 
                    if k not in ['predictions', 'ground_truths', 'confidences']
                }
            
            json.dump(serializable_results, f, indent=2)
        
        # Generate comparison plot
        self._plot_comparison(results, output_path / f"comparison_{timestamp}.png")
        
        # Generate confusion matrix for each model (if ground truth available)
        for model_name, result in results.items():
            if model_name == 'best_model':
                continue
            
            if result.get('ground_truths') and result.get('predictions'):
                self._plot_confusion_matrix(
                    result['ground_truths'],
                    result['predictions'],
                    output_path / f"confusion_{model_name}_{timestamp}.png"
                )
        
        print(f"📊 Report saved to: {output_path}")
        return results_file
    
    def _plot_comparison(self, results: Dict, output_path: Path):
        """Create comparison visualization"""
        models = []
        accuracies = []
        speeds = []
        
        for model_name, result in results.items():
            if model_name == 'best_model' or 'accuracy' not in result:
                continue
            
            if result['accuracy'] is not None:
                models.append(model_name)
                accuracies.append(result['accuracy'])
                speeds.append(1 / result['avg_inference_time'] if result['avg_inference_time'] > 0 else 0)
        
        if not models:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy bar chart
        bars = ax1.bar(models, accuracies)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1%}', ha='center', va='bottom')
        
        # Speed comparison (images per second)
        ax2.bar(models, speeds)
        ax2.set_title('Model Speed Comparison')
        ax2.set_ylabel('Images per second')
        ax2.set_xlabel('Higher is better')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred, output_path: Path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=self.categories)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

def quick_compare(image_folder: str, num_images: int = 20) -> Dict:
    """Quick comparison without ground truth"""
    evaluator = ModelEvaluator(test_data_dir=image_folder)
    test_images = evaluator._get_test_images(Path(image_folder), num_images)
    
    results = {}
    for model_name, model in evaluator.models.items():
        try:
            # Just get confidence and speed
            confidences = []
            times = []
            
            for img_path in test_images[:5]:  # Test only 5 for speed
                import time
                start = time.time()
                result = model.classify_image(str(img_path))
                times.append(time.time() - start)
                confidences.append(result["confidence"])
            
            results[model_name] = {
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "avg_time": np.mean(times) if times else 0,
                "tested_images": len(test_images[:5])
            }
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results