"""
FastAPI backend for Dental X-Ray Classification System.
Provides REST API endpoints for prediction, bulk upload, and retraining.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import io
import zipfile
import shutil
from pathlib import Path
from PIL import Image
import uvicorn
import asyncio
import json
from datetime import datetime
import logging

from src.prediction import DentalPredictor
from src.model import DentalClassifier, DentalTrainer
from src.preprocessing import create_data_loaders

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dental X-Ray Classification API",
    description="AI-powered dental X-ray classification system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
retraining_status = {
    "status": "idle",
    "message": "No retraining in progress",
    "progress": 0
}

# Model paths
MODEL_PATH = "models/dental_model.pth"
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
NEW_DATA_DIR = DATA_DIR / "new"


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global predictor
    
    try:
        # Create directories
        Path("models").mkdir(exist_ok=True)
        NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load model if exists
        if Path(MODEL_PATH).exists():
            predictor = DentalPredictor(MODEL_PATH)
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️ No trained model found. Please train a model first.")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dental X-Ray Classification API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "no_model",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict dental condition from X-ray image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with confidence scores
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        result = predictor.predict(image)
        
        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/upload_bulk")
async def upload_bulk_data(file: UploadFile = File(...), class_name: str = "Unknown"):
    """
    Upload training data - supports both ZIP files and individual images.
    
    For ZIP files:
    - Expected structure: class_name1/image1.jpg, class_name2/image2.jpg
    
    For individual images:
    - Provide class_name parameter to specify the class
    - Image will be saved to: data/new/upload_timestamp/class_name/
    
    Args:
        file: ZIP file or image file (JPG, PNG, JPEG)
        class_name: Class name for individual images (ignored for ZIP files)
        
    Returns:
        Upload status and extraction details
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Create upload directory
        upload_dir = NEW_DATA_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = 0
        class_counts = {}
        
        # Check if it's a ZIP file
        if file.filename.endswith('.zip'):
            # Handle ZIP file
            with zipfile.ZipFile(io.BytesIO(contents), 'r') as zip_ref:
                zip_ref.extractall(upload_dir)
            
            # Count extracted files
            for class_dir in upload_dir.iterdir():
                if class_dir.is_dir():
                    files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
                    class_counts[class_dir.name] = len(files)
                    total_files += len(files)
            
            logger.info(f"Extracted {total_files} images from ZIP to {upload_dir}")
            
        elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Handle individual image
            class_dir = upload_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            image_path = class_dir / file.filename
            with open(image_path, 'wb') as f:
                f.write(contents)
            
            total_files = 1
            class_counts[class_name] = 1
            
            logger.info(f"Saved image to {image_path}")
        else:
            raise HTTPException(
                status_code=400, 
                detail="File must be a ZIP archive or image file (JPG, PNG)"
            )
        
        return {
            "message": f"Successfully uploaded {total_files} image(s)",
            "extraction_path": str(upload_dir.name),
            "class_counts": class_counts,
            "total_files": total_files,
            "file_type": "zip" if file.filename.endswith('.zip') else "image"
        }
        
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with new data.
    
    Returns:
        Retraining status
    """
    global retraining_status
    
    if retraining_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    # Check if new data exists
    if not any(NEW_DATA_DIR.iterdir()):
        raise HTTPException(status_code=400, detail="No new training data found")
    
    # Start retraining in background
    background_tasks.add_task(retrain_model)
    
    retraining_status = {
        "status": "running",
        "message": "Retraining started",
        "progress": 0
    }
    
    return {"message": "Retraining started successfully"}


@app.get("/retrain_status")
async def get_retrain_status():
    """Get current retraining status."""
    return retraining_status


async def retrain_model():
    """Background task for model retraining."""
    global predictor, retraining_status
    
    try:
        logger.info("🔄 Starting model retraining...")
        
        # Update status
        retraining_status.update({
            "status": "running",
            "message": "Preparing data...",
            "progress": 10
        })
        
        # Merge new data with existing training data
        merge_training_data()
        
        retraining_status.update({
            "message": "Loading data...",
            "progress": 30
        })
        
        # Create data loaders
        train_loader, val_loader, class_names = create_data_loaders(
            train_dir=str(TRAIN_DIR),
            batch_size=16  # Smaller batch for retraining
        )
        
        retraining_status.update({
            "message": "Training model...",
            "progress": 50
        })
        
        # Create and train model
        model = DentalClassifier(num_classes=len(class_names), pretrained=True)
        trainer = DentalTrainer(model, class_names=class_names)
        
        # Train for fewer epochs during retraining
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,
            lr=0.0001  # Lower learning rate for fine-tuning
        )
        
        retraining_status.update({
            "message": "Saving model...",
            "progress": 90
        })
        
        # Reload predictor with new model
        predictor = DentalPredictor(MODEL_PATH)
        
        retraining_status.update({
            "status": "completed",
            "message": "Retraining completed successfully",
            "progress": 100
        })
        
        logger.info("✅ Model retraining completed")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {str(e)}")
        retraining_status.update({
            "status": "failed",
            "message": f"Retraining failed: {str(e)}",
            "progress": 0
        })


def merge_training_data():
    """Merge new uploaded data with existing training data."""
    logger.info("📂 Merging training data...")
    
    # Iterate through new data directories
    for upload_dir in NEW_DATA_DIR.iterdir():
        if upload_dir.is_dir():
            # Copy each class directory
            for class_dir in upload_dir.iterdir():
                if class_dir.is_dir():
                    target_class_dir = TRAIN_DIR / class_dir.name
                    target_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all images
                    for img_file in class_dir.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            target_file = target_class_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{img_file.name}"
                            shutil.copy2(img_file, target_file)
    
    # Clean up new data directory after merging
    for upload_dir in NEW_DATA_DIR.iterdir():
        if upload_dir.is_dir():
            shutil.rmtree(upload_dir)
    
    logger.info("✅ Data merging completed")


@app.get("/model_info")
async def get_model_info():
    """Get information about the current model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "class_names": predictor.class_names,
        "device": str(predictor.device),
        "model_path": MODEL_PATH
    }


@app.get("/data_stats")
async def get_data_statistics():
    """Get training data statistics."""
    if not TRAIN_DIR.exists():
        return {"message": "No training data found"}
    
    stats = {}
    total_images = 0
    
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            stats[class_dir.name] = count
            total_images += count
    
    return {
        "total_images": total_images,
        "class_distribution": stats,
        "num_classes": len(stats)
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )