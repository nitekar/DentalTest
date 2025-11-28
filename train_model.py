"""
Quick training script to create a basic model for the dental classification system.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import DentalClassifier, DentalTrainer
from src.preprocessing import create_data_loaders

def quick_train():
    """Train a basic model quickly."""
    print("Starting quick training for dental X-ray classification...")
    
    # Check if data exists
    train_dir = "data/train"
    if not Path(train_dir).exists():
        print("ERROR: Training data not found. Please run organize_data.py first.")
        return
    
    # Configuration
    BATCH_SIZE = 16  # Smaller batch size for quick training
    EPOCHS = 5       # Fewer epochs for quick training
    LEARNING_RATE = 0.001
    
    try:
        # Create data loaders
        print("Loading data...")
        train_loader, val_loader, class_names = create_data_loaders(
            train_dir=train_dir,
            batch_size=BATCH_SIZE,
            num_workers=0  # Avoid multiprocessing issues on Windows
        )
        
        print(f"Classes: {class_names}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Create model
        print("Creating model...")
        model = DentalClassifier(num_classes=len(class_names), pretrained=True)
        
        # Create trainer
        trainer = DentalTrainer(model, class_names=class_names)
        
        # Quick training
        print("Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=3
        )
        
        print("Training completed successfully!")
        print(f"Model saved to: models/dental_model.pth")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_train()