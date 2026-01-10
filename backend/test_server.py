"""
Simplified backend to test startup
"""
from fastapi import FastAPI
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model classes (same as before)
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=14, pretrained=False):
        super(DiseaseClassifier, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)

class BodyPartClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super(BodyPartClassifier, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
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
        return self.fc(x)

class FractureDetector(nn.Module):
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

models_dict = {}

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models_dict
    
    try:
        logger.info("Loading models...")
        
        # Stage A
        logger.info("Loading Stage A...")
        stage_a = DiseaseClassifier(num_classes=1)
        stage_a.load_state_dict(torch.load("stageA_10.pth", map_location=device))
        stage_a.to(device)
        stage_a.eval()
        models_dict['stage_a'] = stage_a
        logger.info("✓ Stage A loaded")
        
        # Stage B
        logger.info("Loading Stage B...")
        stage_b = DiseaseClassifier(num_classes=2)
        stage_b.load_state_dict(torch.load("stageB_10.pth", map_location=device))
        stage_b.to(device)
        stage_b.eval()
        models_dict['stage_b'] = stage_b
        logger.info("✓ Stage B loaded")
        
        # Body
        logger.info("Loading Body Part...")
        body = BodyPartClassifier(num_classes=3)
        body.load_state_dict(torch.load("body_model.pth", map_location=device))
        body.to(device)
        body.eval()
        models_dict['body'] = body
        logger.info("✓ Body Part loaded")
        
        # Fracture
        logger.info("Loading Fracture...")
        fracture = FractureDetector(num_classes=1)
        fracture.load_state_dict(torch.load("fracture_model.pth", map_location=device))
        fracture.to(device)
        fracture.eval()
        models_dict['fracture'] = fracture
        logger.info("✓ Fracture loaded")
        
        logger.info(f"All {len(models_dict)} models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    return {
        "status": "online",
        "models_loaded": len(models_dict),
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
