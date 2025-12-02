import torch
import torch.nn as nn
from torchsummary import summary


import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用简单的CNN 网络不复杂可以防止一下过拟合
# 53% ~ 51.5 %


class SimpleRetinaCNN(nn.Module):
    def __init__(self, num_classes=5):  # RetinaMNIST 5分类
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)   # 28 → 14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)   # 14 → 7

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # 7×7 → 1×1

        self.fc = nn.Linear(128, num_classes)  # 分类头

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# net = SimpleRetinaCNN()
# summary(net, input_size=(3, 28, 28))