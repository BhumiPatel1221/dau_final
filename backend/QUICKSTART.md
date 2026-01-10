# 🚀 Quick Start Guide - MedivisionAI

## Backend is Ready!

Your backend server is **already running** and fully operational with all 4 trained models loaded.

## Current Status
✅ Server: **RUNNING** on http://localhost:8000  
✅ Models: **4/4 loaded** (Stage A, Stage B, Body Part, Fracture)  
✅ Device: **CPU** (GPU auto-detected if available)  
✅ Grad-CAM: **Enabled**  

## Test the Backend

### 1. Health Check
Open your browser and go to:
```
http://localhost:8000
```

You should see:
```json
{
  "status": "online",
  "service": "MedivisionAI Backend",
  "models_loaded": 4,
  "device": "cpu"
}
```

### 2. Check Models Info
```
http://localhost:8000/api/models/info
```

### 3. Test with Frontend

#### Option A: Start the Frontend (Recommended)
```bash
# Open a NEW terminal
cd D:\CascadeProjects\frontend
npm run dev
```

Then open your browser to the URL shown (usually http://localhost:5173)

#### Option B: Test with curl/PowerShell
```powershell
# Create a test request
$file = "path\to\your\xray.jpg"
$uri = "http://localhost:8000/api/analyze"

# Upload and analyze
Invoke-RestMethod -Uri $uri -Method Post -Form @{
    file = Get-Item $file
    mode = "unified"
}
```

## What the Backend Does

When you upload an X-ray image:

1. **Preprocesses** the image (resize, normalize)
2. **Runs 4 models**:
   - Stage A: Disease probability
   - Stage B: Normal vs Abnormal classification
   - Body Part: Identifies body region
   - Fracture: Detects fractures
3. **Generates Grad-CAM** heatmap showing affected areas
4. **Returns JSON** with all predictions and visualizations

## Example Response

```json
{
  "prediction": "Abnormality Detected",
  "confidence": 0.87,
  "gradcam_image": "data:image/png;base64,iVBORw0KG...",
  "body_part_detection": {
    "body_part": "Chest",
    "confidence": 0.95
  },
  "fracture_detection": {
    "fracture_detected": false,
    "label": "No Fracture",
    "confidence": 0.92
  },
  "all_probabilities": {
    "Normal": 0.13,
    "Abnormal": 0.87
  }
}
```

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Restarting the Server

```bash
cd D:\CascadeProjects\backend
python main.py
```

Or use the batch file:
```bash
start_server.bat
```

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill the process if needed
taskkill /PID <process_id> /F
```

### Models not loading
- Ensure all `.pth` files are in the backend directory
- Check Python version (3.8+)
- Reinstall dependencies: `pip install -r requirements.txt`

### CORS errors from frontend
- Verify backend is running on port 8000
- Check frontend's `VITE_API_BASE_URL` environment variable

## Next Steps

1. ✅ Backend is running
2. 🔄 Start the frontend (`cd frontend && npm run dev`)
3. 🖼️ Upload an X-ray image
4. 🎉 See real AI predictions with Grad-CAM visualization!

---

**Need help?** Check `INTEGRATION_SUMMARY.md` for detailed documentation.
