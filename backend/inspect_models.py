"""
Model Architecture Inspector
Inspects saved PyTorch models to determine their architecture
"""

import torch
import sys

def inspect_model(model_path):
    """Inspect a PyTorch model file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check if it's a full model or just state dict
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                print("Format: Full checkpoint with 'state_dict' key")
                state_dict = state_dict['state_dict']
            else:
                print("Format: State dictionary")
        else:
            print("Format: Full model object")
            state_dict = state_dict.state_dict()
        
        # Analyze layers
        print(f"\nTotal parameters: {len(state_dict)}")
        print("\nFirst 10 layers:")
        for i, (name, param) in enumerate(list(state_dict.items())[:10]):
            print(f"  {name}: {param.shape}")
        
        print("\nLast 10 layers:")
        for name, param in list(state_dict.items())[-10:]:
            print(f"  {name}: {param.shape}")
        
        # Determine architecture
        print("\n" + "="*60)
        print("Architecture Analysis:")
        print("="*60)
        
        # Check for common architectures
        if any('densenet' in k.lower() or 'denseblock' in k.lower() for k in state_dict.keys()):
            print("✓ Detected: DenseNet architecture")
            # Find output layer
            for name, param in state_dict.items():
                if 'classifier' in name and 'weight' in name:
                    num_classes = param.shape[0]
                    print(f"  Number of classes: {num_classes}")
                    print(f"  Input features: {param.shape[1]}")
        
        elif any('resnet' in k.lower() or 'layer4' in k.lower() for k in state_dict.keys()):
            print("✓ Detected: ResNet architecture")
            for name, param in state_dict.items():
                if 'fc.weight' in name or ('classifier.weight' in name):
                    num_classes = param.shape[0]
                    print(f"  Number of classes: {num_classes}")
                    print(f"  Input features: {param.shape[1]}")
        
        else:
            print("? Unknown architecture")
            # Try to find final layer
            for name, param in list(state_dict.items())[-5:]:
                if 'weight' in name and len(param.shape) == 2:
                    print(f"  Possible output layer: {name}")
                    print(f"    Output size: {param.shape[0]}")
                    print(f"    Input size: {param.shape[1]}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    models = [
        "stageA_10.pth",
        "stageB_10.pth",
        "body_model.pth",
        "fracture_model.pth"
    ]
    
    for model_path in models:
        try:
            inspect_model(model_path)
        except FileNotFoundError:
            print(f"\n{'='*60}")
            print(f"File not found: {model_path}")
            print(f"{'='*60}")
