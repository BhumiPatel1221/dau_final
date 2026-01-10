import torch

models_info = {
    'stageA_10.pth': 'Stage A',
    'stageB_10.pth': 'Stage B',
    'body_model.pth': 'Body Part',
    'fracture_model.pth': 'Fracture'
}

for path, name in models_info.items():
    try:
        sd = torch.load(path, map_location='cpu')
        
        # Check for classifier
        if 'classifier.weight' in sd:
            num_classes = sd['classifier.weight'].shape[0]
            print(f"{name}: {num_classes} classes (using 'classifier')")
        elif 'fc.weight' in sd:
            num_classes = sd['fc.weight'].shape[0]
            print(f"{name}: {num_classes} classes (using 'fc')")
        else:
            print(f"{name}: Unknown output layer")
            
    except Exception as e:
        print(f"{name}: Error - {e}")
