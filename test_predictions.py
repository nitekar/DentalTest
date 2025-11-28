"""
Quick test script for dental X-ray predictions.
Tests the API with sample images from the test dataset.
"""

import requests
from pathlib import Path
import json
from PIL import Image
import sys

API_URL = "http://localhost:8000"

def test_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy")
            print(f"   Model loaded: {data['model_loaded']}")
            if data['model_loaded']:
                print(f"   Classes: {', '.join(data.get('classes', []))}")
            return True
        else:
            print("❌ API returned error")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the API is running: uvicorn main:app --reload --port 8000")
        return False


def predict_image(image_path):
    """Send image for prediction."""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


def test_single_image(image_path):
    """Test prediction on a single image."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n📸 Testing image: {image_path.name}")
    print(f"   Path: {image_path}")
    
    # Get image info
    try:
        img = Image.open(image_path)
        print(f"   Size: {img.size[0]}x{img.size[1]}")
        print(f"   Mode: {img.mode}")
    except:
        pass
    
    print("\n🔮 Making prediction...")
    result = predict_image(image_path)
    
    if result:
        print("\n✅ PREDICTION RESULTS:")
        print("=" * 60)
        print(f"   🎯 Prediction: {result['prediction']}")
        print(f"   📊 Confidence: {result['confidence']*100:.2f}%")
        print("\n   📈 All Probabilities:")
        
        # Sort by probability
        probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in probs:
            bar = "█" * int(prob * 50)
            print(f"      {class_name:15s} {prob*100:6.2f}% {bar}")
        print("=" * 60)


def test_multiple_images(directory, max_count=5):
    """Test predictions on multiple images from a directory."""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    # Get all image files
    image_files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
    
    if not image_files:
        print(f"❌ No images found in {directory}")
        return
    
    print(f"\n📁 Testing {min(len(image_files), max_count)} images from: {directory}")
    print("=" * 60)
    
    results = []
    
    for i, img_path in enumerate(image_files[:max_count], 1):
        print(f"\n[{i}/{min(len(image_files), max_count)}] {img_path.name}")
        result = predict_image(img_path)
        
        if result:
            print(f"   ✅ {result['prediction']} ({result['confidence']*100:.1f}%)")
            results.append({
                'file': img_path.name,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
        else:
            print(f"   ❌ Failed")
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"   Total tested: {len(results)}")
        print(f"   Average confidence: {avg_confidence*100:.2f}%")
        
        # Count predictions
        pred_counts = {}
        for r in results:
            pred_counts[r['prediction']] = pred_counts.get(r['prediction'], 0) + 1
        
        print("\n   Predictions breakdown:")
        for pred, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {pred}: {count} images")


def test_by_class():
    """Test one image from each class in test directory."""
    test_dir = Path('data/test')
    
    if not test_dir.exists():
        print("❌ data/test directory not found")
        return
    
    print("\n🧪 Testing one image from each class")
    print("=" * 60)
    
    for class_dir in sorted(test_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if images:
                print(f"\n📂 Class: {class_dir.name}")
                result = predict_image(images[0])
                
                if result:
                    pred = result['prediction']
                    conf = result['confidence'] * 100
                    
                    if pred == class_dir.name or pred.replace('_', ' ').lower() == class_dir.name.lower():
                        print(f"   ✅ CORRECT: {pred} ({conf:.1f}%)")
                    else:
                        print(f"   ❌ WRONG: Predicted {pred} (Expected {class_dir.name}) ({conf:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("🦷 Dental X-Ray Prediction Tester")
    print("=" * 60)
    
    # Check API health
    if not test_api_health():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "file":
            if len(sys.argv) > 2:
                test_single_image(sys.argv[2])
            else:
                print("Usage: python test_predictions.py file <image_path>")
        
        elif sys.argv[1] == "dir":
            if len(sys.argv) > 2:
                directory = sys.argv[2]
                max_count = int(sys.argv[3]) if len(sys.argv) > 3 else 5
                test_multiple_images(directory, max_count)
            else:
                print("Usage: python test_predictions.py dir <directory> [max_count]")
        
        elif sys.argv[1] == "class":
            test_by_class()
        
        else:
            print("Unknown command. Usage:")
            print("  python test_predictions.py file <image_path>")
            print("  python test_predictions.py dir <directory> [max_count]")
            print("  python test_predictions.py class")
    
    else:
        # Interactive mode
        print("\nChoose test mode:")
        print("1. Test single image")
        print("2. Test all images in a directory")
        print("3. Test one image from each class")
        print()
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "1":
            path = input("Enter image path: ").strip()
            test_single_image(path)
        
        elif choice == "2":
            directory = input("Enter directory path: ").strip()
            count = input("Max images to test (default 5): ").strip()
            max_count = int(count) if count else 5
            test_multiple_images(directory, max_count)
        
        elif choice == "3":
            test_by_class()
        
        else:
            print("❌ Invalid choice!")
