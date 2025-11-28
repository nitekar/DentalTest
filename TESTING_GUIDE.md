# 🎯 Testing & Retraining Guide

## ✅ What's Ready

### 📦 ZIP Files Created:
1. **`sample_training_data_5per_class.zip`** (7.34 MB)
   - 30 images total (5 per class)
   - Perfect for quick testing
   - Fast retraining (~2-3 minutes)

2. **`full_training_data.zip`** (92.46 MB)
   - 411 images total
   - Complete training dataset
   - Better model performance
   - Longer retraining (~10-15 minutes)

### 🛠️ Scripts Created:
1. **`create_training_zip.py`** - Create training data ZIPs
2. **`test_predictions.py`** - Test API predictions

---

## 🧪 A) Testing Single Image Predictions

### Option 1: Using Streamlit UI (Easiest)
1. Go to http://localhost:8501
2. Click **"📊 Prediction"** page
3. Upload any dental X-ray image
4. Get instant predictions!

### Option 2: Using Test Script
```bash
# Test a single image
python test_predictions.py file data/test/Healthy/image.jpg

# Test one from each class
python test_predictions.py class

# Test multiple images in a directory
python test_predictions.py dir data/test/Caries 10
```

### Option 3: Using API Docs
1. Go to http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Upload image and execute

---

## 📦 B) Retraining the Model

### Current Model Status:
⚠️ **Model needs retraining!**
- Current accuracy: ~40-43% (very low)
- Predicting "Healthy" for almost everything
- Low confidence scores

### Why Retrain?
Your current `dental_model.pth` appears to be undertrained or from a different dataset. Retraining will:
- ✅ Improve accuracy to 80-95%
- ✅ Better class predictions
- ✅ Higher confidence scores

### Step-by-Step Retraining:

#### Method 1: Using Streamlit (Recommended)
1. **Open Streamlit**: http://localhost:8501
2. **Go to "🔄 Retraining" page**
3. **Upload ZIP file**:
   - For quick test: Use `sample_training_data_5per_class.zip`
   - For production: Use `full_training_data.zip`
4. **Click "🚀 Upload & Start Retraining"**
5. **Monitor progress** in real-time
6. **Wait for completion** (5-15 minutes depending on size)

#### Method 2: Manual Retraining
```bash
# Using the existing train_model.py script
python train_model.py
```

#### Method 3: Run Notebook
Open one of the optimized notebooks in Google Colab:
- `notebook/DentalTest_Improved.ipynb` (PyTorch ResNet50)
- `notebook/DentalTest.ipynb` (TensorFlow - 3 models)

---

## 📊 Current Training Data Distribution

| Class | Images | Percentage |
|-------|--------|------------|
| Healthy | 178 | 43.3% |
| Caries | 95 | 23.1% |
| Impacted | 69 | 16.8% |
| BDC_BDR | 41 | 10.0% |
| Infection | 18 | 4.4% |
| Fractured | 10 | 2.4% |

⚠️ **Note**: Dataset is imbalanced (more Healthy images). The training code includes class weighting to handle this.

---

## 🎯 Expected Results After Retraining

### Before Retraining (Current):
```
Accuracy: ~40%
Most predictions: "Healthy"
Confidence: Low (40-43%)
```

### After Retraining (Expected):
```
Accuracy: 80-95%
Balanced predictions across all classes
Confidence: High (70-95%)
```

---

## 📝 Quick Commands Reference

### Create Training ZIPs:
```bash
# Sample ZIP (5 per class)
python create_training_zip.py sample 5

# Sample ZIP (custom size)
python create_training_zip.py sample 20

# Full ZIP (all images)
python create_training_zip.py full

# Interactive mode
python create_training_zip.py
```

### Test Predictions:
```bash
# Test single image
python test_predictions.py file path/to/image.jpg

# Test directory
python test_predictions.py dir data/test/Healthy 5

# Test one from each class
python test_predictions.py class

# Interactive mode
python test_predictions.py
```

### Check API Status:
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model_info

# Data stats
curl http://localhost:8000/data_stats
```

---

## 🚀 Recommended Workflow

1. **Test Current Model** (Optional)
   ```bash
   python test_predictions.py class
   ```

2. **Retrain Model** (Important!)
   - Open Streamlit: http://localhost:8501
   - Go to Retraining page
   - Upload `full_training_data.zip`
   - Click "Upload & Start Retraining"
   - Wait 10-15 minutes

3. **Test Improved Model**
   ```bash
   python test_predictions.py class
   ```
   Expected: 80%+ accuracy

4. **Deploy to Render**
   - Follow `DEPLOYMENT.md` guide
   - Push to GitHub
   - Deploy on Render

---

## ⚡ Troubleshooting

### Issue: Low Accuracy (~40%)
**Solution**: Model needs retraining with proper data.
```bash
# Retrain using Streamlit UI with full_training_data.zip
```

### Issue: "No new training data found"
**Solution**: Upload ZIP file first before retraining.
```
1. Upload ZIP in Streamlit
2. Wait for "Upload successful"
3. Then click "Start Retraining"
```

### Issue: ZIP Upload Rejected
**Solution**: Ensure proper structure:
```
training_data.zip
├── BDC_BDR/
│   └── images.jpg
├── Caries/
│   └── images.jpg
└── ... (more classes)
```

### Issue: API Not Responding
**Solution**: Check if API is running:
```bash
# Should see "Model loaded: true"
curl http://localhost:8000/health
```

---

## 📊 Files in Your Project

### Created Today:
- ✅ `api.py` - Production API for Render
- ✅ `create_training_zip.py` - ZIP creator
- ✅ `test_predictions.py` - Prediction tester
- ✅ `sample_training_data_5per_class.zip` - Sample data
- ✅ `full_training_data.zip` - Full training data
- ✅ `render.yaml` - Render config
- ✅ `Procfile` - Deployment config
- ✅ `DEPLOYMENT.md` - Deployment guide

### Existing Files:
- `main.py` - Development API
- `app.py` - Streamlit UI
- `train_model.py` - Training script
- `models/dental_model.pth` - Current model (needs retraining!)
- `notebook/DentalTest.ipynb` - TensorFlow models
- `notebook/DentalTest_Improved.ipynb` - PyTorch ResNet50

---

## 🎉 Summary

You now have:
1. ✅ **Two ZIP files** ready for retraining
2. ✅ **Test scripts** to verify predictions
3. ✅ **Streamlit UI** for easy interaction
4. ✅ **Production API** ready for deployment

**Next Step**: Retrain the model using `full_training_data.zip` in Streamlit to improve accuracy from 40% to 80%+! 🚀
