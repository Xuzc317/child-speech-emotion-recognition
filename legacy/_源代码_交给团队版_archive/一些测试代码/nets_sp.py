import torch
import torch.nn as nn
import math

import torch.nn.functional as F

# nets_sp.py
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self, input_dim=94, num_classes=6):
        super().__init__()
        self.model = nn.Sequential(
            # 第1层卷积 (B,1,94) -> (B,128,94)
            nn.Conv1d(1, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),  # (B,128,23)
            
            # 第2层卷积 (B,128,23) -> (B,256,23)
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (B,256,11)
            
            # 第3层卷积 (B,256,11) -> (B,512,11)
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # (B,512,16)
            
            # 分类头
            nn.Flatten(),
            nn.Linear(512 * 16, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    """仅残差结构"""
    def __init__(self, in_c, out_c, kernel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel, stride, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel, 1, padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        return nn.ReLU()(x + residual)

class ResCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # 初始卷积层 (B,1,162) -> (B,64,162)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4)  # (B,64,40)
        )
        
        # 残差块堆叠
        self.res_blocks = nn.Sequential(
            ResBlock(64, 128, 5, stride=2),  # (B,128,20)
            ResBlock(128, 256, 3),          # (B,256,20)
            ResBlock(256, 512, 3, stride=2)  # (B,512,10)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B,512,1)
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        return self.classifier(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class SECNN(nn.Module):
    def __init__(self, input_dim=94, num_classes=6):  # 修改1：添加构造函数参数
        super().__init__()
        self.features = nn.Sequential(
            # 修改2：使用传入的input_dim
            nn.Conv1d(1, 128, 7, padding=3),  # 输入通道固定为1
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SEBlock(256),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SEBlock(512),
            nn.AdaptiveAvgPool1d(16)
        )
        # 修改3：使用传入的num_classes
        self.classifier = nn.Linear(512 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)