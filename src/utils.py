import os
import json
import shutil
import zipfile

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from config import OUTPUT_DIR, CATEGORIES

def sort_images_to_folders(classifications: List[Dict], output_name: str = None) -> Path:
    """
    Sort images into category folders based on classifications
    
    Args:
        classifications: List of dicts with keys: 
                         image_path, category, confidence, model
        output_name: Name for output zip file
    
    Returns:
        Path to created zip file
    """
    if not classifications:
        raise ValueError("No classifications provided")
    
    # Create timestamp if no output name
    if not output_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = classifications[0].get('model', 'unknown').lower()
        output_name = f"sorted_{model_name}_{timestamp}.zip"
    
    output_path = OUTPUT_DIR / output_name
    
    # Create temporary working directory
    temp_dir = OUTPUT_DIR / "temp_sorting"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    # Create category folders
    for category in CATEGORIES.keys():
        (temp_dir / category).mkdir(exist_ok=True)
    
    # Create uncertain folder for low confidence
    (temp_dir / "uncertain").mkdir(exist_ok=True)
    (temp_dir / "errors").mkdir(exist_ok=True)
    
    # Statistics
    stats = {
        "total": len(classifications),
        "sorted": 0,
        "uncertain": 0,
        "errors": 0,
        "by_category": {cat: 0 for cat in CATEGORIES.keys()}
    }
    
    # Copy images to appropriate folders
    print("📂 Sorting images into folders...")
    for item in tqdm(classifications, desc="Sorting"):
        image_path = Path(item.get('image_path', ''))
        
        if not image_path.exists():
            stats["errors"] += 1
            continue
        
        # Determine destination
        confidence = item.get('confidence', 0)
        category = item.get('category', '')
        
        if 'error' in item:
            dest_folder = temp_dir / "errors"
            stats["errors"] += 1
        elif confidence < 0.6:  # Confidence threshold
            dest_folder = temp_dir / "uncertain"
            stats["uncertain"] += 1
        elif category in CATEGORIES:
            dest_folder = temp_dir / category
            stats["by_category"][category] += 1
            stats["sorted"] += 1
        else:
            dest_folder = temp_dir / "uncertain"
            stats["uncertain"] += 1
        
        # Copy file with unique name if duplicate exists
        dest_file = dest_folder / image_path.name
        counter = 1
        while dest_file.exists():
            stem = image_path.stem
            suffix = image_path.suffix
            dest_file = dest_folder / f"{stem}_{counter}{suffix}"
            counter += 1
        
        try:
            shutil.copy2(image_path, dest_file)
            
            # Save metadata for each image
            metadata = {
                "original_path": str(image_path),
                "category": category,
                "confidence": confidence,
                "model": item.get('model', 'unknown'),
                "timestamp": datetime.now().isoformat()
            }
            
            metadata_file = dest_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error copying {image_path.name}: {e}")
            stats["errors"] += 1
    
    # Create README file with statistics
    create_readme(temp_dir, stats, classifications[0].get('model', 'unknown'))
    
    # Create zip file
    print(f"📦 Creating zip archive: {output_path.name}")
    create_zip_from_folder(temp_dir, output_path)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print(f"✅ Sorting complete!")
    print(f"   Total images: {stats['total']}")
    print(f"   Successfully sorted: {stats['sorted']}")
    print(f"   Uncertain (low confidence): {stats['uncertain']}")
    print(f"   Errors: {stats['errors']}")
    
    for category, count in stats['by_category'].items():
        if count > 0:
            print(f"   {category}: {count}")
    
    return output_path

def create_zip_from_folder(folder_path: Path, output_zip: Path) -> Path:
    """Create zip file from folder contents"""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(folder_path)
                zipf.write(file_path, arcname)
    
    return output_zip

def create_readme(folder_path: Path, stats: Dict, model_name: str):
    """Create README file with sorting information"""
    readme_content = f"""# ISplatter - Sorted Images

## Sorting Information
- Model used: {model_name}
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total images processed: {stats['total']}

## Statistics
- Successfully sorted: {stats['sorted']} ({stats['sorted']/stats['total']:.1%})
- Uncertain (confidence < 60%): {stats['uncertain']}
- Errors: {stats['errors']}

## Category Distribution
"""
    
    for category, count in stats['by_category'].items():
        if count > 0:
            percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
            readme_content += f"- {category}: {count} ({percentage:.1f}%)\n"
    
    readme_content += f"""
## Folders
1. Category folders: Contains images sorted by predicted category
2. uncertain/: Images with prediction confidence below 60%
3. errors/: Images that couldn't be processed

## Notes
- Each image has a corresponding .json file with metadata
- Original filenames are preserved when possible
- Duplicate names get a numerical suffix
"""
    
    readme_path = folder_path / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

def batch_classify_and_sort(
    classifier,
    image_folder: str,
    output_name: str = None,
    confidence_threshold: float = 0.6
) -> Path:
    """
    Complete pipeline: classify images and sort them
    
    Args:
        classifier: Instance of classifier (CLIP, Custom, or Hybrid)
        image_folder: Path to folder with images to sort
        output_name: Output zip filename
        confidence_threshold: Minimum confidence for sorting
    
    Returns:
        Path to created zip file
    """
    image_folder = Path(image_folder)
    
    # Get all image files
    image_files = []
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for ext in extensions:
        image_files.extend(image_folder.rglob(f"*{ext}"))
        image_files.extend(image_folder.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"🔍 Found {len(image_files)} images to classify")
    
    # Classify all images
    classifications = []
    for img_path in tqdm(image_files, desc="Classifying"):
        try:
            result = classifier.classify_image(str(img_path))
            result['image_path'] = str(img_path)
            
            # Apply confidence threshold
            if result['confidence'] < confidence_threshold:
                result['category'] = 'uncertain'
            
            classifications.append(result)
        except Exception as e:
            print(f"Error classifying {img_path.name}: {e}")
            classifications.append({
                'image_path': str(img_path),
                'category': 'error',
                'confidence': 0.0,
                'error': str(e),
                'model': classifier.__class__.__name__
            })
    
    # Sort into folders and create zip
    return sort_images_to_folders(classifications, output_name)

def list_files_in_folder(folder_path: str, max_files: int = None) -> List[Path]:
    """List image files in a folder"""
    folder = Path(folder_path)
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    all_files = []
    for ext in extensions:
        all_files.extend(folder.rglob(f"*{ext}"))
        all_files.extend(folder.rglob(f"*{ext.upper()}"))
    
    if max_files and len(all_files) > max_files:
        import random
        all_files = random.sample(all_files, max_files)
    
    return all_files

def cleanup_temp_files():
    """Clean up temporary directories"""
    temp_dirs = [
        OUTPUT_DIR / "temp_sorting",
        OUTPUT_DIR / "temp_processing"
    ]
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Cleaned up: {temp_dir}")