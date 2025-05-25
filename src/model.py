from torchvision import models
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def get_model(num_classes=7, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_training_setup(model, lr=0.01, wd=1e-3, ss=8):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=ss, gamma=0.5)
    return criterion, optimizer, scheduler
