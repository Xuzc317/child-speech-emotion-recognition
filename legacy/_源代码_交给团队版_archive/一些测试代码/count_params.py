import torch
from nets import DrseCNN, Transformer  # 导入你已有的模型定义

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔍 模型：{model.__class__.__name__}")
    print(f"➡️ 总参数量：{total_params:,}")
    print(f"✅ 可训练参数量：{trainable_params:,}")
    print("=" * 40)

if __name__ == '__main__':
    # 记得使用相同的类别数
    drse_model = DrseCNN(num_classes=6)
    transformer_model = Transformer(feat_dim=162, num_classes=6)

    count_parameters(drse_model)
    count_parameters(transformer_model)
