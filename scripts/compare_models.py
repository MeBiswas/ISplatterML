#!/usr/bin/env python3
import time
import json
from tqdm import tqdm
from pathlib import Path

from src.clip_classifier import CLIPClassifier
from src.custom_classifier import CustomClassifier
from src.hybrid_classifier import HybridClassifier

def compare_on_dataset(image_folder, num_samples=50):
    """Quick comparison on a subset of images"""
    image_folder = Path(image_folder)
    image_files = list(image_folder.rglob("*.[pj][np]g"))
    image_files.extend(image_folder.rglob("*.webp"))
    
    if len(image_files) > num_samples:
        import random
        image_files = random.sample(image_files, num_samples)
    
    results = {}
    
    # Test CLIP
    print("Testing CLIP...")
    clip_model = CLIPClassifier()
    clip_times = []
    for img_path in tqdm(image_files[:10]):  # Test fewer for speed
        start = time.time()
        result = clip_model.classify_image(str(img_path))
        clip_times.append(time.time() - start)
    
    results['clip'] = {
        'avg_time': sum(clip_times) / len(clip_times),
        'sample_predictions': [clip_model.classify_image(str(img_path)) 
                              for img_path in image_files[:3]]
    }
    
    # Test Custom
    print("\nTesting Custom Model...")
    custom_model = CustomClassifier()
    custom_times = []
    for img_path in tqdm(image_files):
        start = time.time()
        result = custom_model.classify_image(str(img_path))
        custom_times.append(time.time() - start)
    
    results['custom'] = {
        'avg_time': sum(custom_times) / len(custom_times),
        'sample_predictions': [custom_model.classify_image(str(img_path)) 
                              for img_path in image_files[:3]]
    }
    
    # Compare predictions
    print("\n🔍 Comparing predictions on first 10 images...")
    disagreements = 0
    for img_path in image_files[:10]:
        clip_pred = clip_model.classify_image(str(img_path))
        custom_pred = custom_model.classify_image(str(img_path))
        
        if clip_pred['category'] != custom_pred['category']:
            disagreements += 1
            print(f"  Disagreement on {img_path.name}:")
            print(f"    CLIP: {clip_pred['category']} ({clip_pred['confidence']:.1%})")
            print(f"    Custom: {custom_pred['category']} ({custom_pred['confidence']:.1%})")
    
    results['disagreement_rate'] = disagreements / 10
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python compare_models.py <image_folder>")
        sys.exit(1)
    
    results = compare_on_dataset(sys.argv[1])
    
    # Save results
    with open("model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Comparison complete!")
    print(f"CLIP avg time: {results['clip']['avg_time']:.2f}s")
    print(f"Custom avg time: {results['custom']['avg_time']:.2f}s")
    print(f"Disagreement rate: {results['disagreement_rate']:.1%}")