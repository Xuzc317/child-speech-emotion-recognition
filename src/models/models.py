import torch
import torch.nn as nn

import torch.nn.functional as F


class _ONNXAdaptiveAvgPool1d(nn.Module):
    """ONNX-compatible equivalent of nn.AdaptiveAvgPool1d.

    Reproduces PyTorch's adaptive pooling exactly: for each output position i,
    pools over [floor(i*L/S), ceil((i+1)*L/S)). Uses only Slice + ReduceMean
    ops (standard ONNX) — the loop unrolls at trace time.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        L = x.shape[-1]
        segments = []
        for i in range(self.output_size):
            start = i * L // self.output_size
            end = ((i + 1) * L + self.output_size - 1) // self.output_size
            segments.append(x[:, :, start:end].mean(dim=-1, keepdim=True))
        return torch.cat(segments, dim=-1)


class _GlobalAvgPool1d(nn.Module):
    """ONNX-compatible global average pooling over the last dimension.

    Equivalent to nn.AdaptiveAvgPool1d(1) but uses mean() which maps to
    the standard ONNX ReduceMean op.
    """
    def forward(self, x):
        return x.mean(dim=-1, keepdim=True)

# 主流模型1
class CRNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # 输入 (B, 1, feat_dim) feat_dim=162
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # x: (B, 1, T) -> (B, C, T')
        x = self.cnn(x)
        # 转为 (B, T', C)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)  # (B, T', 256)
        out = out[:, -1, :]    # 取最后时刻
        return self.classifier(out)

# 主流模型2
# Total parameters: 21,320,326
# Val Loss: 1.7926 | Acc: 15.36%
# class TransformerModel(nn.Module):
#     def __init__(self, feat_dim=162, num_classes=6, seq_len=100, d_model=512, nhead=8, num_layers=8):
#         super(TransformerModel, self).__init__()

#         self.input_fc = nn.Linear(feat_dim, d_model)
#         self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=1536,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         # x shape: (B, T, feat_dim)
#         x = self.input_fc(x)                           # → (B, T, d_model)
#         x = x + self.pos_embedding[:, :x.size(1), :]   # → (B, T, d_model)
#         x = self.transformer(x)                        # → (B, T, d_model)
#         x = x.mean(dim=1)                              # → (B, d_model)
#         return self.classifier(x)                      # → (B, num_classes)



#Small小样本适配的 Transformer 模型,因为样本量太小了所以需要做模型轻量化调整，否则仍会因参数太多而过拟合。
# Total parameters: 434,694
# Val Loss: 1.5134 | Acc: 62.17%
# class Transformer(nn.Module):
#     def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, hidden_dim=512, dropout=0.3):
#         super(Transformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, hidden_dim)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,  # 增加前馈网络的维度
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, 128),  # 增加隐藏层的维度
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.embedding(x)  # [B, T, H]
#         x = self.transformer_encoder(x)  # [B, T, H]
#         x = x.mean(dim=1)  # 全局平均池化
#         return self.classifier(x)

#Total parameters: 486,534
#Best validation accuracy: 69.48%
class Transformer(nn.Module):
    def __init__(self, input_dim=162, num_classes=6, num_heads=16, num_layers=6, hidden_dim=1024, dropout=0.3):
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


# 主流模型3
class TDNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, dilation=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(),
            _GlobalAvgPool1d()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # x: (B,1,T)
        x = self.tdnn(x)       # (B,256,1)
        x = x.squeeze(-1)      # (B,256)
        return self.classifier(x)


class CNNModel(nn.Module):
    def __init__(self, num_classes=6, feat_dim=162):
        super().__init__()

        self.feat_dim = feat_dim

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
            nn.MaxPool1d(5, 2),

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
            nn.MaxPool1d(5, 2),
            nn.Dropout(0.4),
        )

        # ONNX-compatible: compute flat_dim dynamically instead of hardcoding
        # so the model adapts to any feat_dim without manual recalculation.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, feat_dim)
            dummy_out = self.feature_extractor(dummy)
            flat_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention module (ONNX-compatible)."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.gap = _GlobalAvgPool1d()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.gap(x).view(b, c)
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

class DrseCNN(nn.Module):
    VERSION = "v0"  # increment when architecture changes

    def __init__(self, num_classes=6):
        super().__init__()

        # Stage 1: wide kernel + residual SE
        self.stage1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResSEBlock(128, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(4, 2),
            nn.Dropout(0.3)
        )

        # Stage 2: deep feature extraction
        self.stage2 = nn.Sequential(
            ResSEBlock(128, 256, kernel_size=5, stride=1, padding=2),
            ResSEBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(4, 2),
            nn.Dropout(0.4)
        )

        # Stage 3: fine-grained capture, reduced output length
        self.stage3 = nn.Sequential(
            ResSEBlock(256, 512, kernel_size=3, stride=1, padding=1),
            _ONNXAdaptiveAvgPool1d(8),
            nn.Dropout(0.5)
        )

        # Classifier: slimmed down to ~1.1M params (was ~8.9M)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)    # (B, 128, 80)
        x = self.stage2(x)    # (B, 256, 39)
        x = self.stage3(x)    # (B, 512, 8)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DrseNNet_new(nn.Module):
    """Evidence-based architecture from C-BESD children's speech statistics.

    Design rationale:
    - F0=291±48 Hz (2x adult): multi-scale frontend (k=3,7,11)
    - F0 within-utt std=53 Hz: SE channel attention for discriminative bands
    - SNR=15.6±6.0 dB (noisy): GELU + BatchNorm for noise robustness
    - Syllable rate=3.5±1.0/s: deeper network (7 ResSE blocks)
    - Duration=2.60±0.65s: progressive channel expansion (128→256→384→512)
    - Feature groups (Mel/MFCC/Chroma/ZCR/RMS): multi-scale kernels parallel
      different correlation scales within/across feature groups

    Target: ~9-10M params for cloud GPU (4090 24G).
    """
    VERSION = "v0"

    def __init__(self, num_classes=6):
        super().__init__()

        # Multi-scale frontend: captures feature interactions at 3 scales
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU()
        )
        self.branch_k7 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU()
        )
        self.branch_k11 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),
            nn.BatchNorm1d(128), nn.GELU()
        )

        # Stage 1: wide receptive field (k=7), children's broad pitch range
        self.stage1 = nn.Sequential(
            ResSEBlock(128, 256, kernel_size=7, stride=1, padding=3),
            ResSEBlock(256, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(4, 2),   # 162 -> 80
            nn.Dropout(0.2),
        )

        # Stage 2: deep extraction, progressive channel growth 256->384
        self.stage2 = nn.Sequential(
            ResSEBlock(256, 384, kernel_size=5, stride=1, padding=2),
            ResSEBlock(384, 384, kernel_size=3, stride=1, padding=1),
            ResSEBlock(384, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(4, 2),   # 80 -> 39
            nn.Dropout(0.3),
        )

        # Stage 3: fine-grained capture
        self.stage3 = nn.Sequential(
            ResSEBlock(384, 512, kernel_size=3, stride=1, padding=1),
            ResSEBlock(512, 512, kernel_size=3, stride=1, padding=1),
            _ONNXAdaptiveAvgPool1d(8),  # 39 -> 8
            nn.Dropout(0.4),
        )

        # Classifier: balanced (~23% of total params)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        f3 = self.branch_k3(x)        # (B, 64, 162)
        f7 = self.branch_k7(x)        # (B, 64, 162)
        f11 = self.branch_k11(x)      # (B, 64, 162)
        x = torch.cat([f3, f7, f11], dim=1)  # (B, 192, 162)
        x = self.fusion(x)            # (B, 128, 162)

        x = self.stage1(x)            # (B, 256, 80)
        x = self.stage2(x)            # (B, 384, 39)
        x = self.stage3(x)            # (B, 512, 8)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DenseModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(162, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(768, 1536),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 1, 162) -> (B, 162)
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
        # Input shape: (B, 1, 162) -> (B, 162, 1)
        x = x.permute(0, 2, 1)
        output, _ = self.rnn(x)  # output shape: (B, 162, 512)
        last_output = output[:, -1, :]  # 取最后时刻的输出
        return self.classifier(last_output)  

# === 优化后的LSTM模型 ===
class OptimizedBiLSTM(nn.Module):
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
            dropout=0.3
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
        # 输入形状: [B, 1, 162]
        x = self.pre_net(x)  # [B, 64, 80]
        x = x.permute(0, 2, 1)  # [B, 80, 64]
        outputs, _ = self.lstm(x)  # [B, 80, 256]
        attn_weights = self.attention(outputs)
        context = torch.sum(attn_weights * outputs, dim=1)
        return self.classifier(context)
    



class CNNBiLSTMModel(nn.Module):
    def __init__(self, num_classes=6, input_dim=162, hidden_size=128, num_layers=2):
        super().__init__()

        # CNN特征提取层
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),  # [B,1,input_dim] -> [B,64,input_dim]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,64,input_dim//2]

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B,128,input_dim//4]

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            _ONNXAdaptiveAvgPool1d(16),  # [B,256,16]
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



#CNN+Transformer
# class ChannelAttention(nn.Module):
#     """通道注意力 + 空间Dropout"""
#     def __init__(self, channel, reduction=16, dropout=0.3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction),
#             nn.ReLU(),
#             nn.Linear(channel // reduction, channel),
#             nn.Sigmoid()
#         )
#         self.dropout = nn.Dropout2d(dropout)  # 空间Dropout

#     def forward(self, x):
#         b, c, l = x.shape
#         avg_y = self.avg_pool(x).view(b, c)
#         max_y = self.max_pool(x).view(b, c)
#         y = self.fc(avg_y + max_y).view(b, c, 1)
#         x = x * y.expand_as(x)
#         return self.dropout(x.unsqueeze(2)).squeeze(2)  # 空间Dropout

# class ResTransformerBlock(nn.Module):
#     """残差 + Transformer 混合模块"""
#     def __init__(self, in_ch, out_ch, kernel_size, stride, dropout=0.2):
#         super().__init__()
#         # 卷积分支
#         self.conv_branch = nn.Sequential(
#             nn.Conv1d(in_ch, out_ch, kernel_size, stride, kernel_size//2),
#             nn.BatchNorm1d(out_ch),
#             nn.GELU(),
#             ChannelAttention(out_ch, dropout=dropout)
#         )
#         # Transformer分支
#         self.transformer = TransformerEncoder(
#             TransformerEncoderLayer(
#                 d_model=out_ch,
#                 nhead=4,
#                 dim_feedforward=4*out_ch,
#                 dropout=dropout,
#                 activation='gelu'
#             ),
#             num_layers=2
#         )
#         self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride) if in_ch != out_ch else nn.Identity()

#     def forward(self, x):
#         residual = self.shortcut(x)
#         # 卷积路径
#         conv_out = self.conv_branch(x)  # [B, C, L]
#         # Transformer需要 [L, B, C] 格式
#         trans_out = self.transformer(conv_out.permute(2, 0, 1))  # [L, B, C]
#         trans_out = trans_out.permute(1, 2, 0)  # [B, C, L]
#         return F.gelu(conv_out + trans_out + residual)

# class HybridCTN(nn.Module):
#     def __init__(self, num_classes=6, input_length=163):
#         super().__init__()
#         self.embed = nn.Sequential(
#             nn.Conv1d(1, 256, 15, padding=7),
#             nn.BatchNorm1d(256),
#             nn.GELU(),
#             nn.MaxPool1d(4, 2)
#         )
        
#         self.block1 = ResTransformerBlock(256, 512, kernel_size=11, stride=1, dropout=0.3)
#         self.pool1 = nn.Sequential(
#             nn.MaxPool1d(4, 2),
#             nn.Dropout(0.4)
#         )
        
#         self.block2 = ResTransformerBlock(512, 1024, kernel_size=7, stride=1, dropout=0.4)
#         self.pool2 = nn.Sequential(
#             nn.AdaptiveAvgPool1d(24),
#             nn.Dropout(0.5)
#         )
        
#         self.transformer = TransformerEncoder(
#             TransformerEncoderLayer(
#                 d_model=1024,
#                 nhead=8,
#                 dim_feedforward=4096,
#                 dropout=0.3,
#                 activation='gelu'
#             ),
#             num_layers=4
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(1024 * 24, 2048),
#             nn.BatchNorm1d(2048),
#             nn.GELU(),
#             nn.AlphaDropout(0.5),  # SELU激活专用Dropout
#             nn.Linear(2048, num_classes)
#         )

#     def forward(self, x):
#         x = self.embed(x)          # [B,256,81]
#         x = self.block1(x)        # [B,512,81]
#         x = self.pool1(x)         # [B,512,40]
#         x = self.block2(x)        # [B,1024,40]
#         x = self.pool2(x)         # [B,1024,24]
        
#         # Transformer处理
#         x = x.permute(2, 0, 1)    # [24,B,1024]
#         x = self.transformer(x)   # [24,B,1024]
#         x = x.permute(1, 0, 2)    # [B,24,1024]
#         x = x.contiguous().view(x.size(0), -1)
#         return self.classifier(x)    

if __name__ == "__main__":
    model = DrseCNN()
    inputs = torch.rand(128, 1, 162)
    model.eval()
    model(inputs)
    # traced_model = torch.jit.trace(model, inputs)
    # torch.jit.save(traced_model, "./drse.jit")

    torch.onnx.export(model, inputs, "./drse.onnx")
    # torch.save(model, "./drse.pt")