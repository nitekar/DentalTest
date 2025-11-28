"""
Script to create a ZIP file from training data for model retraining.
Creates a properly structured ZIP file that can be uploaded via the Streamlit UI.
"""

import zipfile
import os
from pathlib import Path
from datetime import datetime

def create_training_zip(source_dir='data/train', output_name=None):
    """
    Create a ZIP file from training data directory.
    
    Args:
        source_dir: Path to training data directory
        output_name: Name for output ZIP file (optional)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ Error: Directory {source_dir} not found!")
        return
    
    # Generate output filename
    if output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'training_data_{timestamp}.zip'
    
    if not output_name.endswith('.zip'):
        output_name += '.zip'
    
    print(f"📦 Creating ZIP file: {output_name}")
    print(f"📁 Source directory: {source_path.absolute()}")
    print("-" * 60)
    
    # Count files by class
    class_counts = {}
    total_files = 0
    
    # Create ZIP file
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for class_dir in source_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_counts[class_name] = 0
                
                # Add all images from this class
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Add to ZIP with structure: class_name/image.jpg
                        arcname = f"{class_name}/{img_file.name}"
                        zipf.write(img_file, arcname)
                        class_counts[class_name] += 1
                        total_files += 1
                
                print(f"✅ {class_name}: {class_counts[class_name]} images")
    
    print("-" * 60)
    print(f"🎉 ZIP file created successfully!")
    print(f"📊 Total: {total_files} images across {len(class_counts)} classes")
    print(f"💾 File: {Path(output_name).absolute()}")
    print(f"📏 Size: {os.path.getsize(output_name) / (1024*1024):.2f} MB")
    print("\n✨ Ready to upload in Streamlit Retraining page!")


def create_sample_zip(sample_size=10):
    """
    Create a small sample ZIP for testing (takes N images from each class).
    
    Args:
        sample_size: Number of images per class to include
    """
    source_path = Path('data/train')
    
    if not source_path.exists():
        print(f"❌ Error: Directory data/train not found!")
        return
    
    output_name = f'sample_training_data_{sample_size}per_class.zip'
    
    print(f"📦 Creating SAMPLE ZIP file: {output_name}")
    print(f"📊 Including {sample_size} images per class")
    print("-" * 60)
    
    class_counts = {}
    total_files = 0
    
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for class_dir in source_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_counts[class_name] = 0
                
                # Get first N images from this class
                images = [f for f in class_dir.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                for img_file in images[:sample_size]:
                    arcname = f"{class_name}/{img_file.name}"
                    zipf.write(img_file, arcname)
                    class_counts[class_name] += 1
                    total_files += 1
                
                print(f"✅ {class_name}: {class_counts[class_name]} images")
    
    print("-" * 60)
    print(f"🎉 Sample ZIP created!")
    print(f"📊 Total: {total_files} images")
    print(f"💾 File: {Path(output_name).absolute()}")
    print(f"📏 Size: {os.path.getsize(output_name) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🦷 Dental Training Data ZIP Creator")
    print("=" * 60)
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "sample":
            sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            create_sample_zip(sample_size)
        elif sys.argv[1] == "full":
            output_name = sys.argv[2] if len(sys.argv) > 2 else None
            create_training_zip(output_name=output_name)
        else:
            print("Usage:")
            print("  python create_training_zip.py sample [N]     - Create sample with N images per class")
            print("  python create_training_zip.py full [name]    - Create full ZIP with all images")
    else:
        # Interactive mode
        print("Choose an option:")
        print("1. Create FULL training data ZIP (all images)")
        print("2. Create SAMPLE ZIP for testing (10 images per class)")
        print("3. Custom sample size")
        print()
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "1":
            create_training_zip()
        elif choice == "2":
            create_sample_zip(10)
        elif choice == "3":
            try:
                size = int(input("Images per class: "))
                create_sample_zip(size)
            except ValueError:
                print("❌ Invalid number!")
        else:
            print("❌ Invalid choice!")
