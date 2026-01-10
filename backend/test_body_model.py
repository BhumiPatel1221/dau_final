"""
Test model loading to debug issues
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BodyPartClassifier(nn.Module):
    """
    Body part detection model based on DenseNet
    This model uses 'fc' layer (saved differently than standard DenseNet)
    """
    def __init__(self, num_classes=3, pretrained=False):
        super(BodyPartClassifier, self).__init__()
        # Load base DenseNet
        densenet = models.densenet121(pretrained=pretrained)
        
        # Extract features (everything except classifier)
        self.features = densenet.features
        self.num_features = 1024  # DenseNet121 output features
        
        # Custom fc layer to match saved model
        self.fc = nn.Linear(self.num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Test loading
print("Creating model...")
model = BodyPartClassifier(num_classes=3)

print("\nModel state dict keys (first 10):")
for i, k in enumerate(list(model.state_dict().keys())[:10]):
    print(f"  {k}")

print("\nModel state dict keys (last 10):")
for k in list(model.state_dict().keys())[-10:]:
    print(f"  {k}")

print("\n" + "="*60)
print("Loading saved model...")
try:
    state_dict = torch.load('body_model.pth', map_location='cpu')
    
    print("\nSaved model keys (first 10):")
    for i, k in enumerate(list(state_dict.keys())[:10]):
        print(f"  {k}")
    
    print("\nSaved model keys (last 10):")
    for k in list(state_dict.keys())[-10:]:
        print(f"  {k}")
    
    print("\n" + "="*60)
    print("Attempting to load state dict...")
    model.load_state_dict(state_dict, strict=False)
    print("✓ Model loaded successfully (strict=False)")
    
    # Check what was loaded
    missing, unexpected = [], []
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Model loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"✗ Strict loading failed: {e}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
