import englishdata as data
from nets import CNNModel, DrseCNN, LSTMModel
import librosa
import torch
import os
import random
from englishdata import extract_features, class_dict

# 数据集路径
DATASET_PATH = r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY'
CHECKPOINT = r'C:\Users\59892\Desktop\毕业\code\best_DrseCNN_model_0.8682.pth'
NUM_CLASS = 6
MODEL = DrseCNN
NUM_TEST_SAMPLES = 1000  # 测试的样本数量

# 标签映射
LABEL_MAP = [
    'ANGER',
    'DISGUST',
    'FEAR',
    'HAPPY',
    'NEUTRAL',
    'SAD'
]

def collect_audio_files(dataset_path):
    """收集所有音频文件路径及其真实标签"""
    audio_files = []
    for emotion in LABEL_MAP:
        emotion_dir = os.path.join(dataset_path, emotion)
        if os.path.exists(emotion_dir):
            for file in os.listdir(emotion_dir):
                if file.endswith('.wav'):
                    audio_files.append((os.path.join(emotion_dir, file), emotion))
    return audio_files

def test_single_file(model, file_path):
    """测试单个音频文件"""
    try:
        data, sampling_rate = librosa.load(file_path, duration=2.5, offset=0.6)
        feature = extract_features(data, sampling_rate)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        result = model(feature)
        res = int(torch.argmax(result[0]))
        return LABEL_MAP[res]
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def test():
    # 加载模型
    model = MODEL(num_classes=NUM_CLASS)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()
    
    # 收集所有音频文件
    all_audio_files = collect_audio_files(DATASET_PATH)
    
    # 随机选择测试样本
    if len(all_audio_files) < NUM_TEST_SAMPLES:
        print(f"Warning: Only {len(all_audio_files)} files found, testing all of them.")
        test_files = all_audio_files
    else:
        test_files = random.sample(all_audio_files, NUM_TEST_SAMPLES)
    
    # 进行测试
    correct = 0
    print(f"{'File':<50} {'True Label':<10} {'Predicted':<10} {'Result':<6}")
    print("-" * 80)
    
    for file_path, true_label in test_files:
        predicted = test_single_file(model, file_path)
        if predicted is None:
            continue
            
        result = "✓" if predicted == true_label else "✗"
        if result == "✓":
            correct += 1
            
        # 只显示文件名，不显示完整路径
        file_name = os.path.basename(file_path)
        print(f"{file_name:<50} {true_label:<10} {predicted:<10} {result:<6}")
    
    # 计算并输出准确率
    accuracy = correct / len(test_files) * 100
    print("\n" + "=" * 40)
    print(f"Tested {len(test_files)} samples")
    print(f"Correct predictions: {correct}/{len(test_files)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 40)

if __name__ == '__main__':
    test()