# MedivisionAI Backend

## Overview
This is the backend server for MedivisionAI, an AI-powered medical X-ray analysis system. It integrates trained deep learning models for disease detection, fracture detection, and body part classification with Grad-CAM explainability.

## Features
- ✅ **Real Model Integration**: Uses your trained PyTorch models (no mock data)
- ✅ **Cascade Inference**: Combines Stage A and Stage B models for improved accuracy
- ✅ **Grad-CAM Visualization**: Generates explainable AI heatmaps showing affected areas
- ✅ **Multi-Model Support**: Disease detection, fracture detection, and body part classification
- ✅ **FastAPI**: High-performance async API server
- ✅ **CORS Enabled**: Works seamlessly with the frontend

## Models Loaded
1. **stageA_10.pth** - Primary disease classifier (ResNet50-based)
2. **stageB_10.pth** - Secondary disease classifier for ensemble
3. **body_model.pth** - Body part detection (7 classes)
4. **fracture_model.pth** - Fracture detection (binary classification)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup
```bash
# Navigate to backend directory
cd D:\CascadeProjects\backend

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

```bash
# Start the FastAPI server
python main.py
```

The server will start on `http://localhost:8000`

### Alternative (using uvicorn directly)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Server Manager (Recommended)
```bash
server_manager.bat
```
Interactive menu to start, stop, restart, and test the server.

## ⚠️ Troubleshooting

### Port 8000 Already in Use
If you see `[Errno 10048] error while attempting to bind on address`, the server is already running.

**Solution:**
```powershell
# Find and kill the process using port 8000
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Then restart
python main.py
```

Or use the server manager:
```bash
server_manager.bat
# Choose option 3: Restart Server
```

See `TROUBLESHOOTING.md` for more details.

## API Endpoints

### 1. Health Check
```
GET /
```
Returns server status and loaded models info.

### 2. Analyze X-ray
```
POST /api/analyze
```
**Parameters:**
- `file` (form-data): X-ray image file (JPEG, PNG, DICOM)
- `mode` (form-data, optional): Analysis mode
  - `unified` (default): All analyses
  - `disease`: Disease detection only
  - `fracture`: Fracture detection only
  - `body_part`: Body part classification only

**Response:**
```json
{
  "prediction": "Pneumonia",
  "confidence": 0.87,
  "all_probabilities": {
    "Pneumonia": 0.87,
    "Normal": 0.08,
    "Atelectasis": 0.03,
    ...
  },
  "gradcam_image": "data:image/png;base64,...",
  "body_part_detection": {
    "body_part": "Chest",
    "confidence": 0.95
  },
  "fracture_detection": {
    "fracture_detected": false,
    "confidence": 0.92,
    "label": "No Fracture"
  },
  "analysis_mode": "unified",
  "image_size": [224, 224]
}
```

### 3. Models Info
```
GET /api/models/info
```
Returns information about loaded models and available labels.

## Model Architecture

### Disease Classifier
- **Base**: ResNet50
- **Classes**: 14 (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Normal)
- **Input**: 224x224 RGB images
- **Preprocessing**: ImageNet normalization

### Cascade Ensemble
The system uses a two-stage cascade approach:
1. **Stage A**: Initial prediction
2. **Stage B**: Refinement prediction
3. **Ensemble**: Averages both predictions for final result

### Grad-CAM
- Generates visual explanations by highlighting regions that influenced the model's decision
- Uses the last convolutional layer (layer4) of ResNet
- Overlays heatmap on original X-ray image

## Image Preprocessing Pipeline

1. **Resize**: 224x224 pixels
2. **Grayscale to RGB**: Convert single-channel to 3-channel
3. **Normalization**: ImageNet mean and std
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## Testing the API

### Using curl
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@path/to/xray.jpg" \
  -F "mode=unified"
```

### Using Python
```python
import requests

url = "http://localhost:8000/api/analyze"
files = {"file": open("xray.jpg", "rb")}
data = {"mode": "unified"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Frontend Integration

The frontend is already configured to call this API:
- **Endpoint**: `/api/analyze`
- **Method**: POST
- **Base URL**: `http://127.0.0.1:8000` (configurable via `VITE_API_BASE_URL`)

## Performance Optimization

- **GPU Acceleration**: Automatically uses CUDA if available
- **Model Caching**: Models loaded once at startup
- **Async Processing**: FastAPI handles concurrent requests efficiently
- **Batch Processing**: Can be extended for batch inference

## Troubleshooting

### Models not loading
- Ensure all `.pth` files are in the `backend` directory
- Check model architecture matches the saved weights
- Verify PyTorch version compatibility

### Out of memory errors
- Reduce batch size (currently 1)
- Use CPU instead of GPU: Set `device = torch.device("cpu")`
- Resize images to smaller dimensions

### CORS errors
- Verify frontend URL is allowed in CORS middleware
- Check that the API base URL in frontend matches backend URL

## Production Deployment

For production, consider:
1. **Environment Variables**: Use `.env` for configuration
2. **HTTPS**: Deploy behind a reverse proxy (nginx)
3. **Authentication**: Add API key or JWT authentication
4. **Rate Limiting**: Prevent abuse
5. **Logging**: Enhanced logging and monitoring
6. **Docker**: Containerize for easy deployment

## License
This project is part of MedivisionAI - AI-powered healthcare early screening system.

## Disclaimer
⚠️ **Medical Disclaimer**: This system is for educational and research purposes only. It does NOT provide medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.
