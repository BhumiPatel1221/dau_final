"""
Test loading all models to identify issues
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=14, pretrained=False):
        super(DiseaseClassifier, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        num_features = 1024
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

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
        x = self.fc(x)
        return x

class FractureDetector(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(FractureDetector, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        num_features = 1024
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

print("="*60)
print("Testing Model Loading")
print("="*60)

# Test Stage A
print("\n1. Testing Stage A (1 class)...")
try:
    model_a = DiseaseClassifier(num_classes=1)
    state_dict = torch.load('stageA_10.pth', map_location='cpu')
    model_a.load_state_dict(state_dict, strict=True)
    print("   ✓ Stage A loaded successfully")
except Exception as e:
    print(f"   ✗ Stage A failed: {e}")

# Test Stage B
print("\n2. Testing Stage B (2 classes)...")
try:
    model_b = DiseaseClassifier(num_classes=2)
    state_dict = torch.load('stageB_10.pth', map_location='cpu')
    model_b.load_state_dict(state_dict, strict=True)
    print("   ✓ Stage B loaded successfully")
except Exception as e:
    print(f"   ✗ Stage B failed: {e}")

# Test Body Part
print("\n3. Testing Body Part (3 classes)...")
try:
    model_body = BodyPartClassifier(num_classes=3)
    state_dict = torch.load('body_model.pth', map_location='cpu')
    model_body.load_state_dict(state_dict, strict=True)
    print("   ✓ Body Part loaded successfully")
except Exception as e:
    print(f"   ✗ Body Part failed: {e}")

# Test Fracture
print("\n4. Testing Fracture (1 class)...")
try:
    model_frac = FractureDetector(num_classes=1)
    state_dict = torch.load('fracture_model.pth', map_location='cpu')
    model_frac.load_state_dict(state_dict, strict=True)
    print("   ✓ Fracture loaded successfully")
except Exception as e:
    print(f"   ✗ Fracture failed: {e}")

print("\n" + "="*60)
print("Testing Complete")
print("="*60)
