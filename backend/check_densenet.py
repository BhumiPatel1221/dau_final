from torchvision import models

for name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
    m = getattr(models, name)()
    print(f'{name}: {m.classifier.in_features} features')
