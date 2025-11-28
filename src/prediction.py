import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import DentalPreprocessor
from src.model import DentalClassifier


class DentalPredictor:
    """Handles model loading and inference for dental X-rays."""
    
    def __init__(self, model_path='models/dental_model.pth', device=None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: torch device (None for auto-detection)
        """
        self.model_path = model_path
        
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load checkpoint first to get class info
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get class names from checkpoint
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        else:
            # Fallback to default
            self.class_names = ['Cavity', 'Fillings', 'Impacted', 'Implant', 'Normal', 'Other']
        
        # Initialize preprocessor
        self.preprocessor = DentalPreprocessor(image_size=224, apply_clahe=True)
        
        # Load model
        self.model = self._load_model(checkpoint)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Classes: {self.class_names}")
    
    def _load_model(self, checkpoint):
        """Load trained model from checkpoint."""
        # Initialize model architecture with correct number of classes
        model = DentalClassifier(num_classes=len(self.class_names), pretrained=False)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def predict(self, image_input):
        """
        Predict class for a single image.
        
        Args:
            image_input: Path to image file, PIL Image, or numpy array
            
        Returns:
            Dictionary with prediction results:
            {
                'prediction': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        # Preprocess image
        image_tensor = self.preprocessor.preprocess_image(
            image_input,
            apply_augmentation=False
        )
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_idx.item()]
        confidence_value = confidence.item()
        
        # Get all class probabilities
        all_probs = {
            self.class_names[i]: probabilities[0][i].item()
            for i in range(len(self.class_names))
        }
        
        result = {
            'prediction': predicted_class,
            'confidence': confidence_value,
            'all_probabilities': all_probs
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img_path in image_paths:
            result = self.predict(img_path)
            result['image_path'] = str(img_path)
            results.append(result)
        
        return results
    
    def predict_top_k(self, image_input, k=3):
        """
        Get top-k predictions for an image.
        
        Args:
            image_input: Image input
            k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, probability)
        """
        # Preprocess image
        image_tensor = self.preprocessor.preprocess_image(
            image_input,
            apply_augmentation=False
        )
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k, dim=1)
        
        top_predictions = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
        
        return top_predictions
    
    def get_feature_maps(self, image_input):
        """
        Extract feature maps for visualization (e.g., GradCAM).
        
        Args:
            image_input: Image input
            
        Returns:
            Feature maps from last convolutional layer
        """
        # Preprocess image
        image_tensor = self.preprocessor.preprocess_image(
            image_input,
            apply_augmentation=False
        )
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Hook to capture feature maps
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook on last conv layer
        handle = self.model.resnet.layer4.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image_tensor)
        
        # Remove hook
        handle.remove()
        
        return features[0] if features else None
    
    def is_model_loaded(self):
        """Check if model is loaded and ready."""
        return self.model is not None


def test_predictor():
    """Test the predictor with sample images."""
    print("Testing Dental Predictor...")
    
    # Initialize predictor
    predictor = DentalPredictor(model_path='models/dental_model.pth')
    
    print(f"\nClass names: {predictor.class_names}")
    print(f"Device: {predictor.device}")
    print(f"Model loaded: {predictor.is_model_loaded()}")
    
    # Test with a sample image (if exists)
    test_image_dir = Path('data/test')
    if test_image_dir.exists():
        # Find first image
        for class_dir in test_image_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                if images:
                    test_image = images[0]
                    print(f"\nTesting with: {test_image}")
                    
                    # Single prediction
                    result = predictor.predict(test_image)
                    print(f"\nPrediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.4f}")
                    print("\nAll probabilities:")
                    for cls, prob in result['all_probabilities'].items():
                        print(f"  {cls}: {prob:.4f}")
                    
                    # Top-3 predictions
                    print("\nTop-3 predictions:")
                    top_3 = predictor.predict_top_k(test_image, k=3)
                    for i, (cls, prob) in enumerate(top_3, 1):
                        print(f"  {i}. {cls}: {prob:.4f}")
                    
                    break
            break
    
    print("\n Predictor test complete!")


if __name__ == "__main__":
    test_predictor()