from torchvision import models
import torch.nn as nn

def ResNet18(weights=None):
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # CIFAR-10 是RGB所以设为 3通道。
    model.fc = nn.Linear(model.fc.in_features, 10)  # 输出类别数设定为10
    return model