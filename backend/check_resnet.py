from torchvision import models

print("ResNet architectures:")
for name in ['resnet18', 'resnet34', 'resnet50']:
    m = getattr(models, name)()
    print(f'  {name}: {m.fc.in_features} features')
