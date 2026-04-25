import os
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np

wav_dirs = r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY'
seed = 2024
class_dict = {
    'ANGER': 0,
    'DISGUST': 1,
    'FEAR': 2,
    'HAPPY': 3,
    'NEUTRAL': 4,
    'SAD': 5
}

class EnglishDataset(Dataset):
    def __init__(self, wav_dir = wav_dirs, max_length=20):
        self.wav_dir = wav_dir
        self.get_list()
        self.max_length = max_length
        data, sampling_rate = librosa.load(os.path.join(wav_dir, self.file_list[0]))
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sampling_rate,
            n_mfcc=40,
            melkwargs={
                'n_fft': 512,
                'win_length': 400,
                'hop_length': 160,
                'n_mels': 40,
                'center': False
            }
        )

    def get_list(self):
        self.file_list = []
        self.label_list = []
        for i in os.listdir(self.wav_dir):
            if os.path.isdir(os.path.join(self.wav_dir, i)):
                for x in os.listdir(os.path.join(self.wav_dir, i)):
                    if x.endswith('.wav'):  # 只添加 .wav 文件
                        self.file_list.append(os.path.join(i, x))
                for x in range(len([f for f in os.listdir(os.path.join(self.wav_dir, i)) if f.endswith('.wav')])):
                    self.label_list.append(class_dict[i])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        wav_path = os.path.join(self.wav_dir, file_id)
        mfcc = get_features(wav_path)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        
        label = torch.tensor(self.label_list[idx])
        return mfcc, label
    
class npyDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.datas = np.load(data_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.datas[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.int64)
    
def collate_fn(batch):
    """处理固定长度的MFCC特征"""
    inputs = torch.stack([item[0] for item in batch])  # (batch_size, 3, 94)
    inputs = inputs.view(inputs.shape[0], 1, -1)  # (batch_size, 1, 282)
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels



def extract_features(data, sample_rate):
        # ZCR # (1,)
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft # (12,)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC # (40,)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value # (1,)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0) 
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram # (40,)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0) 
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result # 总维度: 1+12+40+1+40=94
    # result = np.zeros(94)  # 初始化为0
    
    # # 只计算 MFCC # (40,)
    # mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    # result[13:53] = mfcc  # MFCC 放在 13-52 位置
    
    # return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation  # 原始特征
    res1 = extract_features(data, sample_rate) # 形状: (94,)
    

    
    # 为了保持输出形状为 (3, 94)，重复原始特征三次
    result = np.array([res1, res1, res1])

    # result = np.array(res1)
    
    # # data with noise 加噪后的特征 → 垂直堆叠
    # noise_data = noise(data)
    # res2 = extract_features(noise_data, sample_rate) 
    # result = np.vstack((result, res2)) # stacking vertically 形状变为 (2,94)
    
    # # data with stretching and pitching 拉伸和变调后的特征 → 继续垂直堆叠
    # new_data = stretch(data)
    # data_stretch_pitch = pitch(new_data, sample_rate)
    # res3 = extract_features(data_stretch_pitch, sample_rate)
    # result = np.vstack((result, res3)) # stacking vertically 最终形状: (3,94)
    

    return result

def noise(data):
    np.save('input1.npy', data)
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    np.save('input_noise.npy', data)
    return data

def stretch(data, rate=0.8):
    data = librosa.effects.time_stretch(data, rate=rate)
    np.save('input_stretch.npy', data)
    return data

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    data = librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)
    np.save('input_pitch.npy', data)
    return data


# if __name__ == "__main__":
#     path = r"C:\Users\59892\Desktop\数据集\BESD\BESD\MY\ANGER\1.EF_12 Angry_1.wav"
#     features = get_features(path)

# englishdata.py 需要添加以下代码来生成训练/测试集
if __name__ == "__main__":
    # 生成完整数据集
    dataset = EnglishDataset()
    all_features = []
    all_labels = []
    for i in range(len(dataset)):
        mfcc, label = dataset[i]
        all_features.append(mfcc.numpy())
        all_labels.append(label.item())
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    np.save('./data_all.npy', all_features)
    np.save('./label_all.npy', all_labels)
    print(f"Generated data_all.npy with shape {all_features.shape} and label_all.npy with shape {all_labels.shape}")
    
    # 随机划分训练集(70%)和测试集(30%)
    np.random.seed(seed)
    indices = np.random.permutation(len(all_features))
    split = int(0.7 * len(all_features))

    train_features = all_features[indices[:split]]
    train_labels = all_labels[indices[:split]]
    test_features = all_features[indices[split:]] 
    test_labels = all_labels[indices[split:]]
    
    # 保存划分后的数据
    os.makedirs('./data', exist_ok=True)
    # np.save('./data/train_features.npy', train_features)
    np.save('./data/train_labels.npy', train_labels)
    # np.save('./data/test_features.npy', test_features)
    np.save('./data/test_labels.npy', test_labels)
    np.save('./data/train_features.npy', all_features[indices[:split]])
    np.save('./data/test_features.npy', all_features[indices[split:]])
    
    # 生成给train_2.py用的数据集
    np.save('data.npy', train_features)
    np.save('label.npy', train_labels)
