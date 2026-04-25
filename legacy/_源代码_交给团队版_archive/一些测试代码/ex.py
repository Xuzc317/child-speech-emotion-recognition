# import torch
# from nets import DrseCNN, ResSEBlock
# import torch.nn as nn

# # 验证ResSEBlock
# block = ResSEBlock(256, 256, kernel_size=5, stride=1, padding=2)
# print(f"ResSEBlock参数量: {sum(p.numel() for p in block.parameters())/1e6:.2f}M")

# # 验证DrseCNN各阶段输出形状
# model = DrseCNN()
# dummy_input = torch.randn(1, 1, 163)  # 输入长度163
# print("\n各阶段输出形状验证:")
# print(f"Stage1输出: {model.stage1(dummy_input).shape}")  # 应得[1,128,80]
# print(f"Stage2输出: {model.stage2(model.stage1(dummy_input)).shape}")  # 应得[1,256,39]
# print(f"Stage3输出: {model.stage3(model.stage2(model.stage1(dummy_input))).shape}")  # 应得[1,512,16]
import torch
import torch.nn as nn
from xr import test_loader, train_loader
from torch.utils.benchmark import Timer
from nets import DrseCNN, CNNModel, CNNBiLSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#计算模型总参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 示例使用
# model = DrseCNN()
# print(f"DrseCNN参数量: {count_parameters(model)/1e6:.2f}M")

#测试轻量化
# model = CNNBiLSTMModel().cuda()
# input = torch.randn(1, 1, 163).cuda()
# t = Timer(stmt="model(input)", globals={"model": model, "input": input})
# print(t.timeit(100))  # 若DrseCNN延迟<CNNBiLSTM的50%，仍可称轻量化

# ​​精度对比测试​


# def test_accuracy(model, loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return correct / total

# if __name__ == '__main__':
#     # 初始化模型
#     drse = DrseCNN().cuda()
#     bilstm = CNNBiLSTMModel().cuda()
    
#     # 加载预训练权重（如果有）
#     # drse.load_state_dict(torch.load('drse.pth'))
    
#     # 测试精度
#     drse_acc = test_accuracy(drse, test_loader)
#     bilstm_acc = test_accuracy(bilstm, test_loader)
    
#     print(f"DrseCNN 测试准确率: {drse_acc:.2%}")
#     print(f"CNNBiLSTM 测试准确率: {bilstm_acc:.2%}")

def test_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model, train_loader, test_loader, epochs=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,weight_decay = 5e-5)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # 验证阶段
        val_acc = test_accuracy(model, test_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}')
    
    return model

if __name__ == '__main__':
    # 初始化模型
    drse = DrseCNN().cuda().to(device)
    bilstm = CNNBiLSTMModel().cuda().to(device)
    cnn = CNNModel().cuda().to(device)
    # 训练模型50个epoch
    print("训练DrseCNN模型...")
    drse = train_model(drse, train_loader, test_loader, epochs=100)

    print("\n训练CNN模型...")
    cnn = train_model(cnn, train_loader, test_loader, epochs=100)   
    
    print("\n训练CNNBiLSTM模型...")
    bilstm = train_model(bilstm, train_loader, test_loader, epochs=100)
    
    # 最终测试精度
    print("\n最终测试精度:")
    drse_acc = test_accuracy(drse, test_loader)
    bilstm_acc = test_accuracy(bilstm, test_loader)
    cnn_acc = test_accuracy(cnn, test_loader)   
    print(f"DrseCNN 测试准确率: {drse_acc:.2%}")
    print(f"CNN 测试准确率: {cnn_acc:.2%}")   
    print(f"CNNBiLSTM 测试准确率: {bilstm_acc:.2%}")
 
    # 保存训练好的模型
    torch.save(drse.state_dict(), 'drse_trained.pth')
    torch.save(cnn.state_dict(), 'cnn_trained.pth')    
    torch.save(bilstm.state_dict(), 'bilstm_trained.pth')