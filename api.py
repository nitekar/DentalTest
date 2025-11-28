"""
Dental X-Ray Classification API - Render Deployment Version
Optimized for cloud deployment with environment configuration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import io
import os
import zipfile
import shutil
from pathlib import Path
from PIL import Image
import uvicorn
import json
from datetime import datetime
import logging

from src.prediction import DentalPredictor
from src.model import DentalClassifier, DentalTrainer
from src.preprocessing import create_data_loaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dental X-Ray Classification API",
    description="AI-powered dental X-ray classification system for detecting dental conditions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - Update with your Streamlit URL after deployment
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "https://*.streamlit.app",
    "*"  # Remove this in production, add specific domains
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
retraining_status = {
    "status": "idle",
    "message": "No retraining in progress",
    "progress": 0,
    "timestamp": None
}

# Configuration from environment variables
PORT = int(os.getenv("PORT", 8000))
MODEL_PATH = os.getenv("MODEL_PATH", "models/dental_model.pth")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TRAIN_DIR = DATA_DIR / "train"
NEW_DATA_DIR = DATA_DIR / "new"
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default


@app.on_event("startup")
async def startup_event():
    """Initialize the model and directories on startup."""
    global predictor
    
    try:
        logger.info("🚀 Starting Dental X-Ray API...")
        
        # Create necessary directories
        Path("models").mkdir(exist_ok=True)
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Created directories: models, {TRAIN_DIR}, {NEW_DATA_DIR}")
        
        # Load model if exists
        if Path(MODEL_PATH).exists():
            logger.info(f"📦 Loading model from {MODEL_PATH}...")
            predictor = DentalPredictor(MODEL_PATH)
            logger.info(f"✅ Model loaded successfully! Classes: {predictor.class_names}")
        else:
            logger.warning(f"⚠️  No trained model found at {MODEL_PATH}. API will run in limited mode.")
            logger.info("💡 Upload a model file or trigger training to enable predictions.")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Shutting down Dental X-Ray API...")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Dental X-Ray Classification API",
        "version": "2.0.0",
        "status": "online",
        "model_loaded": predictor is not None,
        "endpoints": {
            "predict": "/predict",
            "bulk_upload": "/upload_bulk",
            "retrain": "/retrain",
            "health": "/health",
            "docs": "/docs"
        },
        "message": "Welcome to the Dental X-Ray Classification API! Visit /docs for documentation."
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy" if predictor is not None else "no_model",
        "model_loaded": predictor is not None,
        "model_path": MODEL_PATH if Path(MODEL_PATH).exists() else None,
        "classes": predictor.class_names if predictor else [],
        "device": str(predictor.device) if predictor else None,
        "timestamp": datetime.now().isoformat(),
        "environment": "production" if os.getenv("RENDER") else "development"
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict dental condition from X-ray image.
    
    Args:
        file: Uploaded image file (JPG, PNG)
        
    Returns:
        {
            "prediction": "Class name",
            "confidence": 0.95,
            "all_probabilities": {"class1": 0.95, "class2": 0.03, ...},
            "timestamp": "2025-11-28T12:00:00"
        }
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure model file exists or train a model first."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image (JPG, PNG)."
        )
    
    try:
        # Read and validate image
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Make prediction
        result = predictor.predict(image)
        result["timestamp"] = datetime.now().isoformat()
        result["filename"] = file.filename
        
        logger.info(
            f"✅ Prediction: {result['prediction']} "
            f"(confidence: {result['confidence']:.2%}) "
            f"for {file.filename}"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict dental conditions for multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    results = []
    errors = []
    
    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            result = predictor.predict(image)
            result["filename"] = file.filename
            result["index"] = idx
            results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Error processing {file.filename}: {str(e)}")
            errors.append({
                "filename": file.filename,
                "index": idx,
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/upload_bulk")
async def upload_bulk_data(file: UploadFile = File(...)):
    """
    Upload bulk training data as ZIP file.
    
    Expected ZIP structure:
    ```
    dataset.zip
    ├── BDC_BDR/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── Caries/
    │   ├── image3.jpg
    │   └── image4.jpg
    └── ...
    ```
    
    Args:
        file: ZIP file with organized training data
        
    Returns:
        Upload status and class distribution
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a ZIP archive (.zip)"
        )
    
    try:
        # Read ZIP file
        contents = await file.read()
        
        if len(contents) > 100 * 1024 * 1024:  # 100MB limit for ZIP
            raise HTTPException(status_code=413, detail="ZIP file too large (max 100MB)")
        
        # Create extraction directory
        extraction_path = NEW_DATA_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        extraction_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 Extracting ZIP to {extraction_path}...")
        
        # Extract ZIP
        with zipfile.ZipFile(io.BytesIO(contents), 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        
        # Count extracted files by class
        total_files = 0
        class_counts = {}
        
        for class_dir in extraction_path.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                image_files = (
                    list(class_dir.glob('*.jpg')) + 
                    list(class_dir.glob('*.jpeg')) + 
                    list(class_dir.glob('*.png'))
                )
                class_counts[class_dir.name] = len(image_files)
                total_files += len(image_files)
        
        logger.info(f"✅ Extracted {total_files} images across {len(class_counts)} classes")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {total_files} images",
            "extraction_path": str(extraction_path.name),
            "class_counts": class_counts,
            "total_files": total_files,
            "num_classes": len(class_counts),
            "timestamp": datetime.now().isoformat()
        }
        
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/retrain")
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    epochs: int = 10,
    learning_rate: float = 0.0001
):
    """
    Trigger model retraining with new data.
    
    Args:
        epochs: Number of training epochs (default: 10)
        learning_rate: Learning rate for training (default: 0.0001)
        
    Returns:
        Retraining task status
    """
    global retraining_status
    
    if retraining_status["status"] == "running":
        raise HTTPException(
            status_code=409, 
            detail="Retraining already in progress. Check /retrain_status for progress."
        )
    
    # Check if new data exists
    if not any(NEW_DATA_DIR.iterdir()):
        raise HTTPException(
            status_code=400, 
            detail="No new training data found. Upload data using /upload_bulk first."
        )
    
    # Validate parameters
    if not (1 <= epochs <= 100):
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 100")
    if not (0.00001 <= learning_rate <= 0.1):
        raise HTTPException(status_code=400, detail="Learning rate must be between 0.00001 and 0.1")
    
    # Start retraining in background
    background_tasks.add_task(retrain_model, epochs=epochs, lr=learning_rate)
    
    retraining_status = {
        "status": "running",
        "message": "Retraining started",
        "progress": 0,
        "timestamp": datetime.now().isoformat(),
        "config": {"epochs": epochs, "learning_rate": learning_rate}
    }
    
    logger.info(f"🔄 Retraining started: {epochs} epochs, lr={learning_rate}")
    
    return {
        "status": "started",
        "message": "Retraining started successfully",
        "check_status_at": "/retrain_status"
    }


@app.get("/retrain_status")
async def get_retrain_status():
    """Get current retraining status and progress."""
    return retraining_status


async def retrain_model(epochs: int = 10, lr: float = 0.0001):
    """Background task for model retraining."""
    global predictor, retraining_status
    
    try:
        logger.info("🔄 Starting model retraining...")
        
        # Step 1: Merge data
        retraining_status.update({
            "status": "running",
            "message": "Merging training data...",
            "progress": 10
        })
        merge_training_data()
        
        # Step 2: Load data
        retraining_status.update({
            "message": "Loading and preparing data...",
            "progress": 30
        })
        
        train_loader, val_loader, class_names = create_data_loaders(
            train_dir=str(TRAIN_DIR),
            batch_size=16
        )
        
        logger.info(f"📊 Training with {len(train_loader.dataset)} samples, {len(class_names)} classes")
        
        # Step 3: Train model
        retraining_status.update({
            "message": f"Training model ({epochs} epochs)...",
            "progress": 50
        })
        
        model = DentalClassifier(num_classes=len(class_names), pretrained=True)
        trainer = DentalTrainer(model, class_names=class_names)
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr
        )
        
        # Step 4: Save and reload
        retraining_status.update({
            "message": "Saving and reloading model...",
            "progress": 90
        })
        
        predictor = DentalPredictor(MODEL_PATH)
        
        # Step 5: Complete
        retraining_status.update({
            "status": "completed",
            "message": "Retraining completed successfully!",
            "progress": 100,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("✅ Model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {str(e)}")
        retraining_status.update({
            "status": "failed",
            "message": f"Retraining failed: {str(e)}",
            "progress": 0,
            "timestamp": datetime.now().isoformat()
        })


def merge_training_data():
    """Merge new uploaded data with existing training data."""
    logger.info("📂 Merging training data...")
    merged_count = 0
    
    for upload_dir in NEW_DATA_DIR.iterdir():
        if upload_dir.is_dir():
            for class_dir in upload_dir.iterdir():
                if class_dir.is_dir():
                    target_class_dir = TRAIN_DIR / class_dir.name
                    target_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all images
                    for img_file in class_dir.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            target_file = target_class_dir / f"{timestamp}_{img_file.name}"
                            shutil.copy2(img_file, target_file)
                            merged_count += 1
    
    # Clean up after merging
    for upload_dir in NEW_DATA_DIR.iterdir():
        if upload_dir.is_dir():
            shutil.rmtree(upload_dir)
    
    logger.info(f"✅ Merged {merged_count} new images into training set")


@app.get("/model_info")
async def get_model_info():
    """Get detailed information about the current model."""
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Upload or train a model first."
        )
    
    return {
        "model_loaded": True,
        "class_names": predictor.class_names,
        "num_classes": len(predictor.class_names),
        "device": str(predictor.device),
        "model_path": MODEL_PATH,
        "model_exists": Path(MODEL_PATH).exists(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/data_stats")
async def get_data_statistics():
    """Get training data statistics."""
    if not TRAIN_DIR.exists():
        return {
            "message": "No training data found",
            "train_dir": str(TRAIN_DIR),
            "total_images": 0
        }
    
    stats = {}
    total_images = 0
    
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            count = (
                len(list(class_dir.glob('*.jpg'))) + 
                len(list(class_dir.glob('*.jpeg'))) + 
                len(list(class_dir.glob('*.png')))
            )
            stats[class_dir.name] = count
            total_images += count
    
    return {
        "total_images": total_images,
        "num_classes": len(stats),
        "class_distribution": stats,
        "train_dir": str(TRAIN_DIR),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/status")
async def api_status():
    """Comprehensive API status for monitoring."""
    return {
        "api": {
            "name": "Dental X-Ray Classification API",
            "version": "2.0.0",
            "status": "operational"
        },
        "model": {
            "loaded": predictor is not None,
            "classes": predictor.class_names if predictor else []
        },
        "data": {
            "train_dir_exists": TRAIN_DIR.exists(),
            "new_data_available": any(NEW_DATA_DIR.iterdir()) if NEW_DATA_DIR.exists() else False
        },
        "retraining": retraining_status,
        "timestamp": datetime.now().isoformat()
    }


# Entry point for Render deployment
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,  # Disable reload in production
        log_level="info"
    )
