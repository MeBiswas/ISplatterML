#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   You have Python {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "transformers",
        "pillow",
        "opencv-python",
        "numpy",
        "tqdm",
        "click",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "albumentations",
    ]
    
    for req in requirements:
        print(f"   Installing {req.split()[0]}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + req.split())
            print(f"   ✅ {req.split()[0]}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {req}")
            return False
    
    return True

def create_directory_structure():
    """Create project directory structure"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data/training/female",
        "data/training/male", 
        "data/training/creatures",
        "data/training/weapons",
        "data/training/fantasy_world",
        "data/test_set",
        "data/unlabeled",
        "models/clip",
        "models/custom",
        "models/hybrid",
        "outputs/clip_sorted",
        "outputs/custom_sorted",
        "outputs/hybrid_sorted",
        "experiments/clip_results",
        "experiments/custom_results",
        "experiments/comparisons",
        "src",
        "scripts",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    return True

def download_clip_model():
    """Download CLIP model weights"""
    print("\n🤖 Testing CLIP import...")
    try:
        # Simple import test. The 'openai-clip' package is just 'clip' when imported.
        import clip
        print("   ✅ CLIP import successful.")
        return True
    except ImportError as e:
        print(f"   ❌ Error importing CLIP: {e}")
        print("   You need to install the correct package:")
        print("   pip install openai-clip")
        return False
    except Exception as e:
        print(f"   ⚠️  Other error with CLIP: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("ISplatter - AI Image Sorting Tool - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directory_structure():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Some packages failed to install.")
        print("   You may need to install them manually")
        cont = input("   Continue anyway? (y/n): ")
        if cont.lower() != 'y':
            sys.exit(1)
    
    # Test CLIP
    download_clip_model()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("\n📚 Next steps:")
    print("   1. Add your training images to data/training/ categories")
    print("   2. Test CLIP immediately: python main.py sort <folder> --model clip")
    print("   3. Train custom model: python main.py train data/training/")
    print("\n💡 Example usage:")
    print('   python main.py sort "C:\\Users\\You\\Pictures\\Fantasy" --model clip')
    print('   python main.py sort "./my_images" --model hybrid')
    print("=" * 60)

if __name__ == "__main__":
    main()