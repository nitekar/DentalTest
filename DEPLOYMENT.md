# 🚀 Deployment Guide - Dental X-Ray Classification API

## Render.com Deployment (Recommended - FREE)

### Prerequisites
1. GitHub account with the DentalTest repository
2. Render.com account (sign up at https://render.com)
3. Trained model file (`dental_model.pth` in `models/` directory)

### Step 1: Prepare Your Repository

1. **Ensure all files are committed:**
   ```bash
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

2. **Required files for Render:**
   - ✅ `api.py` - Main API file (production-ready)
   - ✅ `requirements.txt` - Python dependencies
   - ✅ `render.yaml` - Render configuration
   - ✅ `Procfile` - Process configuration
   - ✅ `runtime.txt` - Python version
   - ✅ `models/dental_model.pth` - Trained model

### Step 2: Deploy on Render

#### Option A: Using render.yaml (Automated)

1. **Go to Render Dashboard:**
   - Visit https://dashboard.render.com
   - Click "New +" → "Blueprint"

2. **Connect GitHub Repository:**
   - Connect your GitHub account
   - Select the `nitekar/DentalTest` repository
   - Render will automatically detect `render.yaml`

3. **Click "Apply":**
   - Render will create the service automatically
   - Wait 5-10 minutes for deployment

#### Option B: Manual Setup

1. **Create New Web Service:**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"

2. **Connect Repository:**
   - Connect GitHub: `nitekar/DentalTest`
   - Branch: `main`

3. **Configure Service:**
   - **Name:** `dental-xray-api`
   - **Region:** Oregon (US West)
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:** 
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn api:app --host 0.0.0.0 --port $PORT
     ```

4. **Environment Variables:**
   - `PYTHON_VERSION`: `3.9.13`
   - `MODEL_PATH`: `models/dental_model.pth`

5. **Plan:**
   - Select "Free" plan (512MB RAM, perfect for testing)

6. **Click "Create Web Service"**

### Step 3: Verify Deployment

Once deployed, your API will be available at:
```
https://dental-xray-api-XXXX.onrender.com
```

Test endpoints:
- Health Check: `https://your-url.onrender.com/health`
- API Docs: `https://your-url.onrender.com/docs`
- Root: `https://your-url.onrender.com/`

### Step 4: Update Streamlit App

Update `app.py` with your Render URL:

```python
# Change this line:
API_URL = "http://localhost:8000"

# To:
API_URL = "https://your-render-url.onrender.com"
```

---

## Alternative Platforms

### Railway.app Deployment

1. **Create Railway Account:** https://railway.app
2. **New Project → Deploy from GitHub**
3. **Select Repository:** `nitekar/DentalTest`
4. **Settings:**
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Add Environment Variables: `MODEL_PATH=models/dental_model.pth`
5. **Deploy**

### Heroku Deployment

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create dental-xray-api

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

---

## 🧪 Local Testing

Before deploying, test locally:

### Terminal 1 - API:
```bash
cd c:\Users\pc\Documents\DentalTest
python api.py
```

### Terminal 2 - Streamlit UI:
```bash
cd c:\Users\pc\Documents\DentalTest
streamlit run app.py
```

Access:
- API: http://localhost:8000
- UI: http://localhost:8501
- Docs: http://localhost:8000/docs

---

## 📊 Monitoring & Logs

### Render Logs:
1. Go to your service dashboard
2. Click "Logs" tab
3. View real-time logs

### Health Check:
Monitor your API at: `https://your-url.onrender.com/health`

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": ["BDC_BDR", "Caries", "Fractured", "Healthy", "Impacted", "Infection"],
  "device": "cpu",
  "timestamp": "2025-11-28T12:00:00"
}
```

---

## 🔧 Troubleshooting

### Issue: Model Not Loading
**Solution:** Ensure `dental_model.pth` is in your Git repository:
```bash
git add models/dental_model.pth
git commit -m "Add trained model"
git push
```

### Issue: Out of Memory
**Solution:** Upgrade to Render's paid plan ($7/month for 2GB RAM)

### Issue: Cold Starts (Free Plan)
**Solution:** Free plans sleep after 15 minutes of inactivity. First request takes ~30 seconds. Upgrade for always-on service.

### Issue: Large Model File
If `dental_model.pth` > 100MB, use Git LFS:
```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes
git add models/dental_model.pth
git commit -m "Track model with Git LFS"
git push
```

---

## 🎯 API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch predictions |
| `/upload_bulk` | POST | Upload training data (ZIP) |
| `/retrain` | POST | Trigger model retraining |
| `/retrain_status` | GET | Check retraining status |
| `/model_info` | GET | Model information |
| `/data_stats` | GET | Training data statistics |
| `/docs` | GET | Interactive API documentation |

---

## 📈 Next Steps

1. ✅ Deploy API to Render
2. ✅ Test API endpoints using /docs
3. ✅ Update Streamlit app with deployed URL
4. ✅ Deploy Streamlit to Streamlit Cloud
5. ✅ Monitor performance and logs

---

## 💡 Tips

- **Free Plan Limitations:** 512MB RAM, sleeps after inactivity
- **Upgrade Benefits:** Always-on, more memory, faster builds
- **Security:** Add authentication for production use
- **CORS:** Update `ALLOWED_ORIGINS` in `api.py` with your domain
- **Monitoring:** Use Render's built-in metrics dashboard

---

## 📞 Support

- Render Docs: https://render.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- GitHub Issues: https://github.com/nitekar/DentalTest/issues

---

**Deployment Checklist:**
- [ ] Model file in repository
- [ ] All dependencies in requirements.txt
- [ ] Environment variables configured
- [ ] Health endpoint working
- [ ] Logs show successful startup
- [ ] Test predictions working
- [ ] Streamlit app connected

🎉 **Ready to deploy!**
