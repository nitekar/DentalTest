"""
Preprocessing module for dental X-ray images.
Handles CLAHE enhancement, augmentation, and data preparation.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import os


class DentalPreprocessor:
    """Handles preprocessing of dental X-ray images."""
    
    def __init__(self, image_size=224, apply_clahe=True):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target image size for model input
            apply_clahe: Whether to apply CLAHE enhancement
        """
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        
        # CLAHE parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def apply_clahe_enhancement(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if len(image.shape) == 3:
            # Convert to grayscale for CLAHE
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply CLAHE
        enhanced = self.clahe.apply(gray)
        
        # Convert back to RGB
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced
    
    def get_train_transforms(self):
        """Get training augmentation transforms."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_val_transforms(self):
        """Get validation transforms (no augmentation)."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_input, apply_augmentation=False):
        """
        Preprocess a single image for inference.
        
        Args:
            image_input: PIL Image, numpy array, or file path
            apply_augmentation: Whether to apply training augmentations
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
            if len(image.shape) == 4:  # RGBA
                image = image[:, :, :3]
        else:
            image = image_input
        
        # Apply CLAHE enhancement
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)
        
        # Choose transforms
        if apply_augmentation:
            transform = self.get_train_transforms()
        else:
            transform = self.get_val_transforms()
        
        # Apply transforms
        transformed = transform(image=image)
        return transformed['image']


class DentalDataset(Dataset):
    """Dataset class for dental X-ray images."""
    
    def __init__(self, data_dir, transform=None, class_names=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to data directory
            transform: Albumentations transform
            class_names: List of class names (auto-detected if None)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get class names
        if class_names is None:
            self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        else:
            self.class_names = class_names
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Collect all image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"Dataset loaded: {len(self.samples)} images, {len(self.class_names)} classes")
    
    def _load_samples(self):
        """Load all image paths and labels."""
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Basic preprocessing
            image = cv2.resize(image, (224, 224))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        class_counts = [0] * len(self.class_names)
        
        for _, label in self.samples:
            class_counts[label] += 1
        
        total_samples = len(self.samples)
        weights = [total_samples / (len(self.class_names) * count) for count in class_counts]
        
        return torch.FloatTensor(weights)


def create_data_loaders(train_dir, val_dir=None, batch_size=32, num_workers=4):
    """
    Create training and validation data loaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory (uses train_dir if None)
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    preprocessor = DentalPreprocessor()
    
    # Create datasets
    train_dataset = DentalDataset(
        train_dir,
        transform=preprocessor.get_train_transforms()
    )
    
    if val_dir and Path(val_dir).exists():
        val_dataset = DentalDataset(
            val_dir,
            transform=preprocessor.get_val_transforms(),
            class_names=train_dataset.class_names
        )
    else:
        # Use training data for validation (with different transforms)
        val_dataset = DentalDataset(
            train_dir,
            transform=preprocessor.get_val_transforms(),
            class_names=train_dataset.class_names
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_names


def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing...")
    
    # Test preprocessor
    preprocessor = DentalPreprocessor()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test preprocessing
    processed = preprocessor.preprocess_image(dummy_image)
    print(f"Processed image shape: {processed.shape}")
    print(f"Processed image dtype: {processed.dtype}")
    print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test with augmentation
    augmented = preprocessor.preprocess_image(dummy_image, apply_augmentation=True)
    print(f"Augmented image shape: {augmented.shape}")
    
    print("✅ Preprocessing test complete!")


if __name__ == "__main__":
    test_preprocessing()