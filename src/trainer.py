# import time
import torch
# import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from .model_builder import build_model
from config import CATEGORIES, MODEL_CONFIGS
from .data_loader import FantasyImageDataset, get_transforms

class ModelTrainer:
    def __init__(self, data_dir: str, model_type: str = "resnet50"):
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create datasets
        train_transform = get_transforms('train')
        val_transform = get_transforms('val')
        
        self.train_dataset = FantasyImageDataset(data_dir, train_transform, 'train')
        self.val_dataset = FantasyImageDataset(data_dir, val_transform, 'val')
        
        # Split datasets
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(self.train_dataset)))
        
        from torch.utils.data import Subset
        self.train_subset = Subset(self.train_dataset, train_indices)
        self.val_subset = Subset(self.val_dataset, val_indices)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_subset, batch_size=32, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_subset, batch_size=32, shuffle=False, num_workers=2
        )
        
        # Build model
        self.model = build_model(model_type, num_classes=len(CATEGORIES))
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Tracking
        self.best_accuracy = 0.0
        self.checkpoint_dir = Path("models/custom")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training on {self.device}")
        print(f"Training samples: {len(self.train_subset)}")
        print(f"Validation samples: {len(self.val_subset)}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'model_type': self.model_type
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🎯 New best model saved with accuracy: {accuracy:.2f}%")
    
    def train(self, epochs: int = 20):
        """Main training loop"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{epochs}:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint if best
            is_best = val_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = val_acc
            
            # Save checkpoint every 5 epochs or if best
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
        
        print(f"\n🎉 Training completed! Best accuracy: {self.best_accuracy:.2f}%")
        return self.best_accuracy

def train_custom_model(data_dir: str, model_type: str = "resnet", epochs: int = 20):
    """Train a custom model and return path to best model"""
    trainer = ModelTrainer(data_dir, model_type)
    best_accuracy = trainer.train(epochs)
    
    best_model_path = Path("models/custom/best_model.pth")
    return best_model_path