import englishdata as data
from nets import CNNModel,DrseCNN,LSTMModel
import librosa
import torch
from englishdata import extract_features, class_dict

# file1 = r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY\DISGUST\1.EF_12 Disgust_3.wav'
# file2 = r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY\HAPPY\68.EM_11 happy_5.wav'
# file3 = r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY\NEUTRAL\54.TF_7_neutral_3.wav'

WAV = r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY\HAPPY\68.EM_11 happy_5.wav'
# WAV = r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY\DISGUST\1.EF_12 Disgust_3.wav' # 待预测的wav文件路径


CHECKPOINT = r'C:\Users\59892\Desktop\毕业\code\best_DrseCNN_model_0.8682.pth' # 保存的模型路径
NUM_CLASS = 6 # 分类类别
MODEL = DrseCNN # 使用的模型
LABEL_MAP = [
    'ANGER',
    'DISGUST',
    'FEAR',
    'HAPPY',
    'NEUTRAL',
    'SAD'
]

def test():
    model = MODEL(num_classes=NUM_CLASS)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()
    # MFCC
    data, sampling_rate = librosa.load(WAV, duration=2.5, offset=0.6)
    feature = extract_features(data, sampling_rate)
    feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    result = model(feature)
    res = int(torch.argmax(result[0]))
    print(LABEL_MAP[res])


if __name__ == '__main__':
    test()