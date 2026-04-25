import torch
import torch.nn as nn
import math
import numpy as np

import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ResSEBlock(nn.Module):
    """带SE注意力的残差块"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.bn2(self.conv2(x))
        x = self.se(x)  # 加入通道注意力
        return F.leaky_relu(x + residual, 0.2)

class DrseNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Stage 1: 高通道数 + 密集卷积
        self.stage1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, padding=3),  # 扩大感受野 1->128 卷积核7*7
            nn.BatchNorm1d(128),    #归一化
            nn.LeakyReLU(0.2),      #激活
            ResSEBlock(128, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(4, 2),  # 输出长度: (163-4)//2 +1 = 80
            nn.Dropout(0.3)
        )
        
        # Stage 2: 深层特征提取
        self.stage2 = nn.Sequential(
            ResSEBlock(128, 256, kernel_size=5, stride=1, padding=2),
            ResSEBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(4, 2),  # 输出长度: (80-4)//2 +1 = 39
            nn.Dropout(0.4)
        )
        
        # Stage 3: 精细特征捕获
        self.stage3 = nn.Sequential(
            ResSEBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(16),  # 统一输出长度为16
            nn.Dropout(0.5)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        np.save('input.npy', x.detach().cpu().numpy())
        x1 = self.stage1(x)    # (batch, 128, 80)
        np.save('t1.npy', x1.detach().cpu().numpy())
        x2 = self.stage2(x1)   # (batch, 256, 39)
        np.save('t2.npy', x2.detach().cpu().numpy())
        x3 = self.stage3(x2)   # (batch, 512, 16)
        np.save('t3.npy', x3.detach().cpu().numpy())
        x_flat = x3.view(x3.size(0), -1)  # Flatten
        return self.classifier(x_flat)
    



#Total parameters: 486,534
#Best validation accuracy: 69.48%
class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=16, num_layers=6, hidden_dim=1024, dropout=0.3):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # 增加前馈网络的维度
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),  # 增加隐藏层的维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)  # [B, T, H]
        x = self.transformer_encoder(x)  # [B, T, H]
        x = x.mean(dim=1)  # 全局平均池化
        return self.classifier(x)




class CNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # 输入形状: (batch, 1, 163)
        self.feature_extractor = nn.Sequential(
            # Stage 1
            nn.Conv1d(1, 256, kernel_size=5, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(5,2),
            
            # Stage 2
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(5,2),
            nn.Dropout(0.4),
            
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        x = self.softmax(x)
        return x


class DenseModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 輸入展平後的維度: 1 * 25 * 11 = 275
        self.feature_extractor = nn.Sequential(
            # 替代Stage 1
            nn.Linear(162, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # 替代Stage 2
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # 替代Stage 3
            nn.Linear(768, 1536),
            nn.ReLU(),
            nn.Dropout(0.5)


        )
        
        # 保持原分類器結構不變（輸入維度需匹配1536）
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)  # 明确指定dim=1，消除警告im=1)  # 明确指定dim=1，消除警告
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平輸入 (batch, 1, 25, 11) -> (batch, 275)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1, 
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, 1, 163) -> (B, 163, 1)
        x = x.permute(0, 2, 1)
        output, _ = self.rnn(x)  # output shape: (B, 163, 512)
        last_output = output[:, -1, :]  # 取最后时刻的输出
        return self.classifier(last_output)  

# === 优化后的LSTM模型 ===
class BiLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # 输入特征处理
        self.pre_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2)
        )
        
        # 核心LSTM结构
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # 输入形状: [B, 1, 163]
        x = self.pre_net(x)  # [B, 64, 54]
        x = x.permute(0, 2, 1)  # [B, 54, 64]
        outputs, _ = self.lstm(x)  # [B, 54, 256]
        attn_weights = self.attention(outputs)
        context = torch.sum(attn_weights * outputs, dim=1)
        return self.classifier(context)
    



class CNNBiLSTMModel(nn.Module):
    def __init__(self, num_classes=6, input_dim=94, hidden_size=128, num_layers=2):
        super().__init__()
        
        # CNN特征提取层
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),  # 输入形状 [B,1,94]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,64,47]
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # [B,128,47]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,128,23]
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # [B,256,23]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # [B,256,16]
            nn.Dropout(0.5)
        )
        
        # BiLSTM时序建模层
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2 * 16, 512),
            nn.AlphaDropout(0.6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        # 调整输入形状
        x = x.view(x.size(0), 1, -1)  # [B,1,input_dim]
        
        # CNN处理
        cnn_out = self.cnn(x)  # [B,256,16]
        
        # 调整维度输入LSTM
        lstm_in = cnn_out.permute(0, 2, 1)  # [B,16,256]
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(lstm_in)  # [B,16,256]
        
        # 拼接时序特征
        features = lstm_out.contiguous().view(lstm_out.size(0), -1)  # [B,16 * 256]
        
        # 分类
        return self.classifier(features)





import torch
import torch.nn as nn

# Squeeze-and-Excitation (SE) Block for 1D
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# Spatial Attention for 1D CNN
class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out) * x

# Temporal Attention for Bi-GRU
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        energy = self.fc(lstm_output).squeeze(2)
        attention_weights = self.softmax(energy)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights



class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out) * x

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        energy = self.fc(lstm_output).squeeze(2)
        attention_weights = self.softmax(energy)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights

class AdaptedSigWavNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=6, 
                 hidden_size_gru=128, num_layers_gru=2, dropout_rate=0.3):
        super(AdaptedSigWavNet, self).__init__()
        
        # 初始特征转换层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 1D CNN处理序列特征
        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2),
            SEBlock1D(128),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2),
            SEBlock1D(256),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2),
            SEBlock1D(512),
            nn.Dropout(dropout_rate)
        )
        
        # Spatial Attention
        self.spatial_attention = SpatialAttention1D(kernel_size=3)

        # Bi-GRU with Temporal Attention
        self.bi_gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size_gru,
            num_layers=num_layers_gru,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers_gru > 1 else 0
        )
        self.temporal_attention = TemporalAttention(hidden_size_gru * 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_gru * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入形状: (batch_size, 1, 163)
        x = self.initial_conv(x)
        
        # 通过卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 应用空间注意力
        x = self.spatial_attention(x)
        
        # 准备GRU输入: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # 双向GRU
        gru_output, _ = self.bi_gru(x)
        
        # 应用时间注意力
        context_vector, _ = self.temporal_attention(gru_output)
        
        # 分类
        output = self.classifier(context_vector)
        return output
    



class CNNBiLSTMModel(nn.Module):
    def __init__(self, num_classes=6, input_dim=94, hidden_size=128, num_layers=2):
        super().__init__()
        
        # CNN特征提取层
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),  # 输入形状 [B,1,94]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,64,47]
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # [B,128,47]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,128,23]
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # [B,256,23]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # [B,256,16]
            nn.Dropout(0.5)
        )
        
        # BiLSTM时序建模层
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2 * 16, 512),
            nn.AlphaDropout(0.6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        # 调整输入形状
        x = x.view(x.size(0), 1, -1)  # [B,1,input_dim]
        
        # CNN处理
        cnn_out = self.cnn(x)  # [B,256,16]
        
        # 调整维度输入LSTM
        lstm_in = cnn_out.permute(0, 2, 1)  # [B,16,256]
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(lstm_in)  # [B,16,256]
        
        # 拼接时序特征
        features = lstm_out.contiguous().view(lstm_out.size(0), -1)  # [B,16 * 256]
        
        # 分类
        return self.classifier(features)





import torch
import torch.nn as nn

# Squeeze-and-Excitation (SE) Block for 1D
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# Spatial Attention for 1D CNN
class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out) * x

# Temporal Attention for Bi-GRU
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        energy = self.fc(lstm_output).squeeze(2)
        attention_weights = self.softmax(energy)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights



class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out) * x

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        energy = self.fc(lstm_output).squeeze(2)
        attention_weights = self.softmax(energy)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights

class AdaptedSigWavNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=6, 
                 hidden_size_gru=128, num_layers_gru=2, dropout_rate=0.3):
        super(AdaptedSigWavNet, self).__init__()
        
        # 初始特征转换层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 1D CNN处理序列特征
        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2),
            SEBlock1D(128),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2),
            SEBlock1D(256),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2),
            SEBlock1D(512),
            nn.Dropout(dropout_rate)
        )
        
        # Spatial Attention
        self.spatial_attention = SpatialAttention1D(kernel_size=3)

        # Bi-GRU with Temporal Attention
        self.bi_gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size_gru,
            num_layers=num_layers_gru,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers_gru > 1 else 0
        )
        self.temporal_attention = TemporalAttention(hidden_size_gru * 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_gru * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入形状: (batch_size, 1, 163)
        x = self.initial_conv(x)
        
        # 通过卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 应用空间注意力
        x = self.spatial_attention(x)
        
        # 准备GRU输入: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # 双向GRU
        gru_output, _ = self.bi_gru(x)
        
        # 应用时间注意力
        context_vector, _ = self.temporal_attention(gru_output)
        
        # 分类
        output = self.classifier(context_vector)
        return output
    



class CNNBiLSTMModel(nn.Module):
    def __init__(self, num_classes=6, input_dim=94, hidden_size=128, num_layers=2):
        super().__init__()
        
        # CNN特征提取层
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),  # 输入形状 [B,1,94]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,64,47]
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # [B,128,47]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,128,23]
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # [B,256,23]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # [B,256,16]
            nn.Dropout(0.5)
        )
        
        # BiLSTM时序建模层
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2 * 16, 512),
            nn.AlphaDropout(0.6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        # 调整输入形状
        x = x.view(x.size(0), 1, -1)  # [B,1,input_dim]
        
        # CNN处理
        cnn_out = self.cnn(x)  # [B,256,16]
        
        # 调整维度输入LSTM
        lstm_in = cnn_out.permute(0, 2, 1)  # [B,16,256]
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(lstm_in)  # [B,16,256]
        
        # 拼接时序特征
        features = lstm_out.contiguous().view(lstm_out.size(0), -1)  # [B,16 * 256]
        
        # 分类
        return self.classifier(features)





import torch
import torch.nn as nn

# Squeeze-and-Excitation (SE) Block for 1D
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# Spatial Attention for 1D CNN
class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out) * x

# Temporal Attention for Bi-GRU
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        energy = self.fc(lstm_output).squeeze(2)
        attention_weights = self.softmax(energy)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights

class SimpleRNNNet(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=6, dropout=0.3):
        super(SimpleRNNNet, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        # x: (batch, seq_len, input_size) or (batch, 1, seq_len, input_size)
        if x.dim() == 4:
            x = x.squeeze(1)  # (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.classifier(out)
        return out
    
if __name__ == "__main__":
    model = DrseNet(num_classes=6).to('cuda')
    x = torch.randn(8, 1, 100, 40)  # (batch, 1, seq_len, input_size)
    path = r"C:\Users\59892\Desktop\数据集\BESD\BESD\MY\ANGER\1.EF_12 Angry_1.wav"
    import librosa
    data, _ = librosa.load(path, sr=16000)
    data = torch.tensor([data]).to('cuda')
    print(data.shape)
    exit()
    sdict = torch.load("best_Drsecnn_model_0.8599.pth", map_location=torch.device('cuda'))
    model.load_state_dict(sdict)
    output = model(data)
    print(output.shape)  # 应该输出 (8, 6)