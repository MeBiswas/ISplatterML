import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_CONFIGS = {
    "clip": {
        "name": "CLIP-ViT-B-32",
        "variant": "ViT-B/32",
        "type": "zero-shot",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
     "custom_resnet": {
        "name": "ResNet50-Custom",
        "base_model": "resnet50",
        "type": "custom",
        "num_classes": 5,
        "checkpoint": MODELS_DIR / "custom" / "resnet_best.pth",
    },
    "custom_efficientnet": {
        "name": "EfficientNet-B0-Custom",
        "base_model": "efficientnet_b0",
        "type": "custom",
        "num_classes": 5,
        "checkpoint": MODELS_DIR / "custom" / "efficientnet_best.pth",
    }
}

# Categories for classification
CATEGORIES = {
    "female": {
        "id": 0,
        "clip_prompts": [
            "a female fantasy character",
            "woman warrior fantasy art",
            "female mage character design",
            "fantasy princess artwork",
            "female elf character"
        ]
    },
    "male": {
        "id": 1,
        "clip_prompts": [
            "a male fantasy character",
            "male warrior fantasy art",
            "fantasy knight character",
            "male wizard artwork",
            "dwarf warrior character"
        ]
    },
    "creatures": {
        "id": 2,
        "clip_prompts": [
            "fantasy creature monster",
            "mythical beast artwork",
            "dragon fantasy art",
            "magical creature design",
            "fantasy animal concept art"
        ]
    },
    "weapons": {
        "id": 3,
        "clip_prompts": [
            "fantasy weapon sword",
            "magical staff artifact",
            "ornate dagger fantasy",
            "bow and arrow fantasy",
            "enchanted weapon artwork"
        ]
    },
    "fantasy_world": {
        "id": 4,
        "clip_prompts": [
            "fantasy landscape scenery",
            "magical forest environment",
            "medieval castle fantasy",
            "dungeon interior concept art",
            "fantasy city landscape"
        ]
    }
}

# CLIP configuration
CLIP_CONFIG = {
    "top_k": 3,
    "temperature": 1.0,
    "normalize": True
}