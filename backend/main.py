"""
MedivisionAI Backend - FastAPI Server
Integrates trained ML models for X-ray disease detection with Grad-CAM explainability
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import base64
import numpy as np
import cv2
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedivisionAI API", version="1.0.0")

# CORS middleware - allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL ARCHITECTURE DEFINITIONS
# ============================================================================

class DiseaseClassifier(nn.Module):
    """
    Disease classification model based on DenseNet architecture
    Matches the saved model structure exactly
    """
    def __init__(self, num_classes=14, pretrained=False):
        super(DiseaseClassifier, self).__init__()
        # Use DenseNet121 as backbone (matches the saved models)
        densenet = models.densenet121(pretrained=pretrained)
        
        # Keep the features part
        self.features = densenet.features
        
        # Modify final layer for our number of classes
        num_features = 1024  # DenseNet121 has 1024 features
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def get_features(self, x):
        """Extract features before final classification layer"""
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return out


class BodyPartClassifier(nn.Module):
    """
    Body part detection model based on ResNet18
    Uses 'fc' layer with 512 input features
    """
    def __init__(self, num_classes=3, pretrained=False):
        super(BodyPartClassifier, self).__init__()
        # Load ResNet18 (512 features)
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract all layers except fc
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Custom fc layer
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FractureDetector(nn.Module):
    """
    Fracture detection model based on DenseNet
    """
    def __init__(self, num_classes=1, pretrained=False):
        super(FractureDetector, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanations
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            cam: Heatmap as numpy array
        """
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = model_output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def apply_gradcam_overlay(original_image: np.ndarray, cam: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Apply Grad-CAM heatmap overlay on original image
    
    Args:
        original_image: Original image as numpy array (H, W, C)
        cam: CAM heatmap (H, W)
        alpha: Overlay transparency
    
    Returns:
        Overlayed image as numpy array
    """
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = (1 - alpha) * original_image + alpha * heatmap
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    
    return overlayed


# ============================================================================
# MODEL LOADING
# ============================================================================

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Disease labels - Stage A has 1 output, Stage B has 2 outputs
# We'll use sigmoid for Stage A (binary) and softmax for Stage B
DISEASE_LABELS_STAGE_A = ["Disease Probability"]  # Single output (0-1)
DISEASE_LABELS_STAGE_B = ["Normal", "Abnormal"]  # Binary classification

BODY_PART_LABELS = [
    "CHEST", "HAND", "SHOULDER"  # 3 classes as per user's code
]

FRACTURE_LABELS = ["Fracture Probability"]  # Single output (0-1)

# Initialize models
models_dict = {}

def load_models():
    """Load all trained models"""
    global models_dict
    
    try:
        # Load Stage A model (1 output - probability)
        logger.info("Loading Stage A disease classifier...")
        stage_a_model = DiseaseClassifier(num_classes=1)
        stage_a_model.load_state_dict(torch.load("stageA_10.pth", map_location=device))
        stage_a_model.to(device)
        stage_a_model.eval()
        models_dict['stage_a'] = stage_a_model
        logger.info("✓ Stage A model loaded (1 output)")
        
        # Load Stage B model (2 outputs - binary classification)
        logger.info("Loading Stage B disease classifier...")
        stage_b_model = DiseaseClassifier(num_classes=2)
        stage_b_model.load_state_dict(torch.load("stageB_10.pth", map_location=device))
        stage_b_model.to(device)
        stage_b_model.eval()
        models_dict['stage_b'] = stage_b_model
        logger.info("✓ Stage B model loaded (2 classes)")
        
        # Load Body Part classifier (3 classes, uses fc layer)
        logger.info("Loading body part classifier...")
        body_model = BodyPartClassifier(num_classes=3)
        body_model.load_state_dict(torch.load("body_model.pth", map_location=device))
        body_model.to(device)
        body_model.eval()
        models_dict['body'] = body_model
        logger.info("✓ Body part model loaded (3 classes)")
        
        # Load Fracture detector (1 output - probability)
        logger.info("Loading fracture detector...")
        fracture_model = FractureDetector(num_classes=1)
        fracture_model.load_state_dict(torch.load("fracture_model.pth", map_location=device))
        fracture_model.to(device)
        fracture_model.eval()
        models_dict['fracture'] = fracture_model
        logger.info("✓ Fracture model loaded (1 output)")
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# Standard ImageNet normalization (commonly used for medical imaging transfer learning)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes: bytes) -> tuple:
    """
    Preprocess uploaded image for model inference
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Tuple of (preprocessed_tensor, original_image_array)
    """
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Store original for Grad-CAM overlay
    original_array = np.array(image)
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor.to(device), original_array


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def chest_pipeline(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Chest X-ray analysis pipeline (matches user's chest_pipeline)
    Stage A: Early abnormal detection (threshold 0.25)
    Stage B: Localized vs Diffuse classification
    """
    with torch.no_grad():
        # Stage A: Early abnormal detection
        stage_a_output = models_dict['stage_a'](image_tensor)
        prob_ab = torch.sigmoid(stage_a_output)[0, 0].item()
        
        # If probability < 0.25, it's NORMAL
        if prob_ab < 0.25:
            return {
                'prediction': 'NORMAL',
                'confidence': 1.0 - prob_ab,
                'explanation': 'No early abnormal lung pattern detected.',
                'stage_a_score': prob_ab,
                'needs_gradcam': False,
                'all_probabilities': {
                    'Normal': 1.0 - prob_ab,
                    'Abnormal': prob_ab,
                    'Disease_Probability_Stage_A': prob_ab
                }
            }
        
        # Stage B: Classify as Localized or Diffuse
        stage_b_output = models_dict['stage_b'](image_tensor)
        probs = F.softmax(stage_b_output, dim=1)[0]
        
        localized = probs[0].item()  # Index 0 = Localized (TB-like)
        diffuse = probs[1].item()    # Index 1 = Diffuse (Pneumonia-like)
        
        # Determine result based on thresholds (0.40)
        if diffuse >= 0.40:
            result = "DIFFUSE abnormal pattern (Pneumonia-like)"
        elif localized >= 0.40:
            result = "LOCALIZED abnormal pattern (TB-like)"
        else:
            result = "Early abnormal lung pattern (uncertain)"
        
        explanation = (
            "Early abnormal lung texture deviation detected. "
            "Highlighted regions show where the model focused."
        )
        
        return {
            'prediction': result,
            'confidence': max(localized, diffuse),
            'explanation': explanation,
            'stage_a_score': prob_ab,
            'localized_prob': localized,
            'diffuse_prob': diffuse,
            'needs_gradcam': True,
            'all_probabilities': {
                'Normal': 1.0 - prob_ab,
                'Abnormal': prob_ab,
                'Localized': localized,
                'Diffuse': diffuse,
                'Disease_Probability_Stage_A': prob_ab
            }
        }


def fracture_pipeline(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Fracture detection pipeline (matches user's fracture_pipeline)
    Threshold: 0.5
    """
    with torch.no_grad():
        output = models_dict['fracture'](image_tensor)
        prob = torch.sigmoid(output)[0, 0].item()
        
        # If probability < 0.5, it's NORMAL
        if prob < 0.5:
            return {
                'prediction': 'NORMAL',
                'confidence': 1.0 - prob,
                'explanation': 'No fracture detected.',
                'fracture_probability': prob,
                'needs_gradcam': False,
                'fracture_detected': False,
                'label': 'No Fracture'
            }
        
        # Fracture detected
        return {
            'prediction': 'FRACTURE DETECTED',
            'confidence': prob,
            'explanation': 'Structural bone irregularity detected.',
            'fracture_probability': prob,
            'needs_gradcam': True,
            'fracture_detected': True,
            'label': 'Fracture Detected'
        }


def detect_body_part(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Step 1: Detect body part (CHEST, ABDOMEN, OTHER)
    """
    with torch.no_grad():
        output = models_dict['body'](image_tensor)
        probs = F.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, 0)
        
        body_part = BODY_PART_LABELS[predicted_idx.item()]
        
        return {
            'body_part': body_part,
            'confidence': float(confidence.cpu().numpy()),
            'index': predicted_idx.item()
        }


def generate_gradcam_visualization(
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    model_name: str = 'stage_a'
) -> str:
    """
    Generate Grad-CAM visualization
    
    Args:
        image_tensor: Preprocessed image tensor
        original_image: Original image array
        model_name: Which model to use for Grad-CAM
    
    Returns:
        Base64 encoded image string
    """
    try:
        model = models_dict[model_name]
        
        # Get the last convolutional layer for DenseNet
        if hasattr(model, 'features'):
            # For DenseNet, use the features layer
            target_layer = model.features[-1]  # Last layer of features
        else:
            logger.warning("Could not find target layer for Grad-CAM")
            return None
        
        # Create Grad-CAM instance
        gradcam = GradCAM(model, target_layer)
        
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor)
        
        # Resize original image to match expected size
        original_resized = cv2.resize(original_image, (224, 224))
        
        # Apply overlay
        overlayed = apply_gradcam_overlay(original_resized, cam, alpha=0.4)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {e}")
        return None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "MedivisionAI Backend",
        "models_loaded": len(models_dict),
        "device": str(device)
    }


@app.post("/api/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    mode: str = Form(default="unified")
):
    """
    Analyze X-ray image following the exact pipeline:
    1. Detect body part
    2. If CHEST -> chest_pipeline (Stage A + B)
    3. If not CHEST -> fracture_pipeline
    4. Generate Grad-CAM only if needed
    
    Args:
        file: Uploaded X-ray image
        mode: Analysis mode (kept for compatibility, but uses pipeline logic)
    
    Returns:
        JSON with prediction results and Grad-CAM visualization
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        image_tensor, original_image = preprocess_image(image_bytes)
        
        # STEP 1: Detect body part
        logger.info("Step 1: Detecting body part...")
        body_part_result = detect_body_part(image_tensor)
        body_part = body_part_result['body_part']
        logger.info(f"Body part detected: {body_part}")
        
        # STEP 2: Run appropriate pipeline based on body part
        if body_part == "CHEST":
            # CHEST pipeline
            logger.info("Step 2: Running CHEST pipeline...")
            result = chest_pipeline(image_tensor)
        else:
            # FRACTURE pipeline (for HAND, SHOULDER)
            logger.info(f"Step 2: Running FRACTURE pipeline for {body_part}...")
            result = fracture_pipeline(image_tensor)
        
        # Add body part detection to result
        result['body_part_detection'] = body_part_result
        
        # STEP 3: Generate Grad-CAM only if needed
        if result.get('needs_gradcam', False):
            logger.info("Step 3: Generating Grad-CAM visualization...")
            
            # Use Stage A model for chest, fracture model for others
            model_name = 'stage_a' if body_part == "CHEST" else 'fracture'
            
            gradcam_image = generate_gradcam_visualization(
                image_tensor, original_image, model_name=model_name
            )
            if gradcam_image:
                result['gradcam_image'] = gradcam_image
        else:
            logger.info("Step 3: No Grad-CAM needed (NORMAL result)")
        
        # Add metadata
        result['analysis_mode'] = mode
        result['image_size'] = original_image.shape[:2]
        result['pipeline'] = 'chest' if body_part == "CHEST" else 'fracture'
        
        # Clean up the needs_gradcam flag (internal use only)
        result.pop('needs_gradcam', None)
        
        logger.info(f"Analysis complete: {result.get('prediction', 'N/A')}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "models": list(models_dict.keys()),
        "stage_a_labels": DISEASE_LABELS_STAGE_A,
        "stage_b_labels": DISEASE_LABELS_STAGE_B,
        "body_part_labels": BODY_PART_LABELS,
        "fracture_labels": FRACTURE_LABELS,
        "device": str(device),
        "model_info": {
            "stage_a": "1 output (sigmoid) - disease probability",
            "stage_b": "2 classes (softmax) - Normal vs Abnormal",
            "body": "3 classes - body part classification",
            "fracture": "1 output (sigmoid) - fracture probability"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
