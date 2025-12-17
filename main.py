#!/usr/bin/env python3
import click
from pathlib import Path, PureWindowsPath, PurePosixPath
import json
from datetime import datetime
import sys
import os

# Add src to path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

try:
    from clip_classifier import CLIPClassifier
    from custom_classifier import CustomClassifier
    from hybrid_classifier import HybridClassifier
    from evaluator import ModelEvaluator, quick_compare
    from utils import batch_classify_and_sort, list_files_in_folder
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you have all the required modules in the src/ directory")
    sys.exit(1)

def normalize_path(path_str: str) -> Path:
    """Normalize path for cross-platform compatibility"""
    # Expand user directory (~)
    if path_str.startswith('~'):
        path_str = os.path.expanduser(path_str)
    
    # Convert to Path object
    path = Path(path_str)
    
    # Resolve relative paths
    if not path.is_absolute():
        path = Path.cwd() / path
    
    return path.resolve()

@click.group()
def cli():
    """ISplatter Hybrid - Compare CLIP vs Custom Training"""
    pass

@cli.command()
@click.argument('image_folder', type=click.Path(exists=False))
@click.option('--model', 
              type=click.Choice(['clip', 'custom', 'hybrid'], case_sensitive=False),
              default='hybrid',
              help='Model to use: clip, custom, or hybrid')
@click.option('--output', default=None, help='Output zip name')
@click.option('--compare', is_flag=True, help='Compare all models')
@click.option('--confidence', default=0.6, type=float, help='Confidence threshold (0.0-1.0)')
def sort(image_folder, model, output, compare, confidence):
    """Sort images using selected model"""
    # Normalize and validate path
    try:
        image_folder_path = normalize_path(image_folder)
    except Exception as e:
        click.echo(f"❌ Error parsing path: {e}")
        click.echo(f"   Provided: {image_folder}")
        click.echo(f"   Current directory: {Path.cwd()}")
        return
    
    # Check if path exists
    if not image_folder_path.exists():
        click.echo(f"❌ Error: Path does not exist: {image_folder_path}")
        click.echo(f"   Please provide a valid path to your images")
        click.echo(f"   Example: python main.py sort \"D:\\My Images\\Fantasy\" --model clip")
        click.echo(f"   Or use relative path: python main.py sort \"./my_images\" --model clip")
        return
    
    if compare:
        # Compare all models
        click.echo(f"🔍 Comparing all models on images from: {image_folder_path}")
        results = quick_compare(str(image_folder_path))
        click.echo("\n📊 Quick Comparison Results:")
        for model_name, result in results.items():
            if 'error' in result:
                click.echo(f"❌ {model_name}: ERROR - {result['error']}")
            else:
                click.echo(f"✅ {model_name}:")
                click.echo(f"   Avg confidence: {result['avg_confidence']:.1%}")
                click.echo(f"   Avg time per image: {result['avg_time']:.3f}s")
                click.echo(f"   Images tested: {result['tested_images']}")
        return
    
    # Use selected model
    click.echo(f"🔍 Loading {model.upper()} model...")
    
    try:
        if model == 'clip':
            classifier = CLIPClassifier()
            suffix = '_clip'
        elif model == 'custom':
            classifier = CustomClassifier()
            suffix = '_custom'
        else:  # hybrid
            classifier = HybridClassifier()
            suffix = '_hybrid'
    except Exception as e:
        click.echo(f"❌ Error loading model: {e}")
        click.echo("Make sure you have all dependencies installed:")
        click.echo("  pip install torch torchvision transformers pillow")
        return
    
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"sorted_images{suffix}_{timestamp}.zip"
    
    click.echo(f"📁 Processing: {image_folder_path}")
    click.echo(f"🎯 Confidence threshold: {confidence:.0%}")
    click.echo(f"📦 Output file: {output}")
    
    try:
        # Process images
        output_path = batch_classify_and_sort(
            classifier=classifier,
            image_folder=str(image_folder_path),
            output_name=output,
            confidence_threshold=confidence
        )
        
        click.echo(f"✅ Done! Output saved to: {output_path}")
        click.echo(f"📁 You can find it in the outputs/ directory")
        
    except Exception as e:
        click.echo(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

@cli.command()
@click.argument('training_data', type=click.Path(exists=False))
@click.option('--epochs', default=20, type=int, help='Training epochs')
@click.option('--model-type', 
              type=click.Choice(['resnet', 'efficientnet'], case_sensitive=False),
              default='resnet',
              help='Base model architecture')
def train(training_data, epochs, model_type):
    """Train a custom model"""
    # Normalize path
    try:
        training_data_path = normalize_path(training_data)
    except Exception as e:
        click.echo(f"❌ Error parsing path: {e}")
        return
    
    # Check if path exists
    if not training_data_path.exists():
        click.echo(f"❌ Error: Training data directory does not exist: {training_data_path}")
        click.echo(f"   Make sure you have organized your training data like this:")
        click.echo(f"   training_data/")
        click.echo(f"   ├── female/")
        click.echo(f"   ├── male/")
        click.echo(f"   ├── creatures/")
        click.echo(f"   ├── weapons/")
        click.echo(f"   └── fantasy_world/")
        return
    
    click.echo(f"🚀 Training {model_type} model with data from: {training_data_path}")
    
    try:
        from trainer import train_custom_model
        model_path = train_custom_model(
            data_dir=str(training_data_path),
            model_type=model_type,
            epochs=epochs
        )
        click.echo(f"🎉 Model saved to: {model_path}")
        click.echo(f"   You can now use: python main.py sort <your_images> --model custom")
    except Exception as e:
        click.echo(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

@cli.command()
@click.argument('test_folder', type=click.Path(exists=False))
@click.option('--ground-truth', type=click.Path(exists=False),
              help='Folder with ground truth labels')
def evaluate(test_folder, ground_truth):
    """Evaluate and compare all models"""
    # Normalize paths
    try:
        test_folder_path = normalize_path(test_folder)
        if ground_truth:
            ground_truth_path = normalize_path(ground_truth)
        else:
            ground_truth_path = None
    except Exception as e:
        click.echo(f"❌ Error parsing path: {e}")
        return
    
    # Check if test folder exists
    if not test_folder_path.exists():
        click.echo(f"❌ Error: Test folder does not exist: {test_folder_path}")
        return
    
    click.echo(f"📊 Evaluating models on: {test_folder_path}")
    
    try:
        evaluator = ModelEvaluator(
            test_data_dir=str(test_folder_path),
            ground_truth_dir=str(ground_truth_path) if ground_truth_path else None
        )
        results = evaluator.compare_all_models()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"model_comparison_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        click.echo(f"📊 Results saved to: {results_file}")
        
        if 'best_model' in results:
            click.echo(f"\n🏆 Best model: {results['best_model']['model_name']}")
            click.echo(f"   Accuracy: {results['best_model']['accuracy']:.1%}")
            click.echo(f"   Avg confidence: {results['best_model']['avg_confidence']:.1%}")
            click.echo(f"   Avg inference time: {results['best_model']['avg_inference_time']:.3f}s")
        
    except Exception as e:
        click.echo(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

@cli.command()
@click.argument('image_file', type=click.Path(exists=False))
@click.option('--model', default='hybrid',
              type=click.Choice(['clip', 'custom', 'hybrid'], case_sensitive=False))
def test(image_file, model):
    """Test a single image with selected model"""
    # Normalize path
    try:
        image_file_path = normalize_path(image_file)
    except Exception as e:
        click.echo(f"❌ Error parsing path: {e}")
        return
    
    # Check if file exists
    if not image_file_path.exists():
        click.echo(f"❌ Error: Image file does not exist: {image_file_path}")
        return
    
    click.echo(f"\n🔬 Testing: {image_file_path.name}")
    click.echo("-" * 40)
    
    try:
        if model == 'clip':
            classifier = CLIPClassifier()
            result = classifier.classify_image(str(image_file_path))
            click.echo(f"🤖 Model: CLIP")
            click.echo(f"🎯 Prediction: {result['category']}")
            click.echo(f"📊 Confidence: {result['confidence']:.1%}")
            
        elif model == 'custom':
            classifier = CustomClassifier()
            result = classifier.classify_image(str(image_file_path))
            click.echo(f"🤖 Model: Custom ResNet")
            click.echo(f"🎯 Prediction: {result['category']}")
            click.echo(f"📊 Confidence: {result['confidence']:.1%}")
            
        else:  # hybrid
            classifier = HybridClassifier()
            result = classifier.classify_image(str(image_file_path))
            click.echo(f"🤖 Model: Hybrid (CLIP + Custom)")
            click.echo(f"🎯 Final Prediction: {result['category']}")
            click.echo(f"📊 Final Confidence: {result['confidence']:.1%}")
            click.echo(f"\n🔍 Individual model predictions:")
            click.echo(f"   CLIP: {result['clip_prediction']} ({result['clip_confidence']:.1%})")
            click.echo(f"   Custom: {result['custom_prediction']} ({result['custom_confidence']:.1%})")
        
        # Show all category probabilities
        click.echo(f"\n📈 All category probabilities:")
        for category, score in result.get('all_scores', {}).items():
            click.echo(f"   {category}: {score:.1%}")
            
    except Exception as e:
        click.echo(f"❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()

@cli.command()
def setup():
    """Setup project directories and check dependencies"""
    from config import DATA_DIR, OUTPUTS_DIR, MODELS_DIR
    
    click.echo("🔧 Setting up ISplatter project...")
    
    # Create directories
    directories = [
        DATA_DIR,
        DATA_DIR / "training",
        DATA_DIR / "test_set",
        DATA_DIR / "unlabeled",
        MODELS_DIR,
        MODELS_DIR / "clip",
        MODELS_DIR / "custom",
        MODELS_DIR / "hybrid",
        OUTPUTS_DIR,
        OUTPUTS_DIR / "clip_sorted",
        OUTPUTS_DIR / "custom_sorted",
        OUTPUTS_DIR / "hybrid_sorted",
        Path("experiments"),
        Path("experiments") / "clip_results",
        Path("experiments") / "custom_results",
        Path("experiments") / "comparisons",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        click.echo(f"   Created: {directory}")
    
    # Check dependencies
    click.echo("\n📦 Checking dependencies...")
    dependencies = ['torch', 'torchvision', 'transformers', 'PIL', 'click']
    
    for dep in dependencies:
        try:
            if dep == 'PIL':
                __import__('PIL')
            else:
                __import__(dep)
            click.echo(f"   ✅ {dep}")
        except ImportError:
            click.echo(f"   ❌ {dep} (missing)")
    
    click.echo(f"\n✅ Setup complete!")
    click.echo(f"\n📚 Next steps:")
    click.echo(f"   1. Organize some training images into:")
    click.echo(f"      data/training/female/, data/training/male/, etc.")
    click.echo(f"   2. Train a model: python main.py train data/training/")
    click.echo(f"   3. Test with CLIP: python main.py sort <your_images> --model clip")

@cli.command()
def demo():
    """Run a quick demo with sample images (if available)"""
    click.echo("🎬 Running ISplatter Demo...")
    
    # Check if we have any images in the current directory
    current_dir = Path.cwd()
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    found_images = []
    
    for ext in image_extensions:
        found_images.extend(current_dir.rglob(f"*{ext}"))
        found_images.extend(current_dir.rglob(f"*{ext.upper()}"))
    
    if len(found_images) == 0:
        click.echo("❌ No images found in current directory.")
        click.echo("   Please add some images and try again.")
        return
    
    # Use first 5 images for demo
    demo_images = found_images[:5]
    click.echo(f"📸 Found {len(demo_images)} images for demo")
    
    # Create a temp directory for demo
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp(prefix="isplatter_demo_"))
    click.echo(f"📁 Using temporary directory: {temp_dir}")
    
    # Copy demo images to temp directory
    for img in demo_images:
        shutil.copy2(img, temp_dir / img.name)
    
    # Test with CLIP
    click.echo("\n🔍 Testing CLIP model...")
    try:
        classifier = CLIPClassifier()
        for img in temp_dir.iterdir():
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                result = classifier.classify_image(str(img))
                click.echo(f"   {img.name}: {result['category']} ({result['confidence']:.0%})")
    except Exception as e:
        click.echo(f"   ❌ Error: {e}")
        click.echo("   Make sure CLIP dependencies are installed")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    click.echo(f"\n✅ Demo complete!")

if __name__ == '__main__':
    # Set up better error handling
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)