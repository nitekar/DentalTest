"""
Script to organize dental X-ray data into proper train/test structure.
"""

import shutil
from pathlib import Path
import random
import os

def organize_dental_data():
    """Organize the dental classification data."""
    
    # Source directory
    source_dir = Path("data/Dental OPG (Classification)")
    
    # Target directories
    train_dir = Path("data/train")
    test_dir = Path("data/test")
    
    # Create target directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Class mapping (clean names)
    class_mapping = {
        "BDC-BDR": "BDC_BDR",
        "Caries": "Caries", 
        "Fractured Teeth": "Fractured",
        "Healthy Teeth": "Healthy",
        "Impacted teeth": "Impacted",
        "Infection": "Infection"
    }
    
    print("Organizing dental X-ray data...")
    
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return
    
    total_images = 0
    
    # Process each class
    for original_class, clean_class in class_mapping.items():
        class_source = source_dir / original_class
        
        if not class_source.exists():
            print(f"WARNING: Class directory not found: {class_source}")
            continue
        
        # Get all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(list(class_source.glob(ext)))
        
        if not images:
            print(f"WARNING: No images found in {class_source}")
            continue
        
        print(f"Processing {clean_class}: {len(images)} images")
        
        # Create class directories
        train_class_dir = train_dir / clean_class
        test_class_dir = test_dir / clean_class
        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        
        # Shuffle images
        random.shuffle(images)
        
        # Split 80/20 train/test
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Copy training images
        for i, img_path in enumerate(train_images):
            target_path = train_class_dir / f"{clean_class}_train_{i:04d}{img_path.suffix}"
            shutil.copy2(img_path, target_path)
        
        # Copy test images
        for i, img_path in enumerate(test_images):
            target_path = test_class_dir / f"{clean_class}_test_{i:04d}{img_path.suffix}"
            shutil.copy2(img_path, target_path)
        
        total_images += len(images)
        print(f"  SUCCESS: Train: {len(train_images)}, Test: {len(test_images)}")
    
    print(f"\nData organization complete!")
    print(f"Total images processed: {total_images}")
    
    # Print summary
    print("\nDataset Summary:")
    for split in ['train', 'test']:
        split_dir = Path(f"data/{split}")
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            total_split = 0
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob('*')))
                    print(f"  {class_dir.name}: {count} images")
                    total_split += count
            print(f"  Total {split}: {total_split} images")

if __name__ == "__main__":
    organize_dental_data()