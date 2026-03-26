import torch.nn as nn

def build_model(num_classes, weights_path=None, device='cpu'):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224*224*3, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model
