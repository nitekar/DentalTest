# Model Upload Guide

Your model file needs to be hosted online so the deployed API can download it.

## Quick Options:

### Option 1: Google Drive (Recommended for quick setup)
1. Upload `models/dental_model.pth` to Google Drive
2. Right-click → Share → Get link → Set to "Anyone with the link"
3. Copy the file ID from the URL (between `/d/` and `/view`)
   - Example URL: `https://drive.google.com/file/d/1ABcD3FgHiJkLmNoPqRsTuVwXyZ/view`
   - File ID: `1ABcD3FgHiJkLmNoPqRsTuVwXyZ`
4. Use this download URL format:
   ```
   https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
   ```

### Option 2: Dropbox
1. Upload `models/dental_model.pth` to Dropbox
2. Get shareable link
3. Change the end from `?dl=0` to `?dl=1`

### Option 3: Hugging Face (Best for ML models)
1. Create account at https://huggingface.co
2. Create a new model repository
3. Upload `dental_model.pth`
4. Use the raw file URL

### Option 4: GitHub Release (For permanent hosting)
1. Go to your repo: https://github.com/nitekar/DentalTest
2. Click "Releases" → "Create a new release"
3. Upload `dental_model.pth` as an asset
4. Use the download URL from the release

## After uploading:

1. Set the MODEL_URL environment variable in your deployment:
   - Render: Dashboard → Environment → Add `MODEL_URL=your_direct_download_url`
   - Heroku: Settings → Config Vars → Add `MODEL_URL`
   
2. Redeploy your application

The API will automatically download the model on startup if it doesn't exist locally.

## Testing locally:

```powershell
# Set environment variable
$env:MODEL_URL="https://your-download-url-here"

# Delete local model to test download
Remove-Item models/dental_model.pth

# Start API
uvicorn main:app --reload

# It should download the model on startup
```
