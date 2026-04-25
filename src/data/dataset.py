import os
import sys
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np

# NOTE: Replace with your own dataset path
# Original path: r'C:\Users\59892\Desktop\数据集\BESD\BESD\MY'
wav_dirs = ''  # Set to your BESD dataset directory
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
        if not wav_dir or not os.path.exists(wav_dir):
            raise ValueError(f"Dataset directory not found: {wav_dir}\n"
                           "Please set the correct path to your BESD dataset.")
        self.get_list()
        self.max_length = max_length

    def get_list(self):
        self.file_list = []
        self.label_list = []
        for i in os.listdir(self.wav_dir):
            if os.path.isdir(os.path.join(self.wav_dir, i)):
                for x in os.listdir(os.path.join(self.wav_dir, i)):
                    self.file_list.append(os.path.join(i, x))
                for x in range(len(os.listdir(os.path.join(self.wav_dir, i)))):
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
    inputs = torch.stack([item[0] for item in batch])  # (batch_size, 1, 25, max_length)
    inputs = inputs.view(inputs.shape[0], 1, inputs.shape[1])
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


def extract_features(data, sample_rate):
    # ZCR (1)
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    # Chroma_stft (12)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC — librosa default n_mfcc=20
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value (1)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram — librosa default n_mels=128
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result  # 1 + 12 + 20 + 1 + 128 = 162

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data, sample_rate)  # (162,)

    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # (2, 162)

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # (3, 162)
    
    return result

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


if __name__ == "__main__":
    if not wav_dirs or not os.path.exists(wav_dirs):
        print("Error: Dataset directory not found.")
        print("Please set the 'wav_dirs' variable at the top of this file to your BESD dataset directory.")
        print("Example: wav_dirs = '/path/to/your/BESD/dataset'")
        sys.exit(1)

    file_list = []
    label_list = []
    a = []
    b = []
    for i in os.listdir(wav_dirs):
        if os.path.isdir(os.path.join(wav_dirs, i)):
            print(i)
            length = 0
            for x in os.listdir(os.path.join(wav_dirs, i)):
                try:
                    sub_dir = os.path.join(i, x)
                    file_path = os.path.join(wav_dirs, sub_dir)
                    mfcc = get_features(file_path)
                    for k in range(3):
                        a.append(mfcc[k])
                        length += 1
                    # a.append(mfcc)
                    # length += 1
                except:
                    pass
            for x in range(length):
                b.append(class_dict[i])

    a = np.array(a)
    b = np.array(b)
    print(a.shape)
    print(b.shape)
    print(b.sum())

    np.save('./data_all.npy', a)
    np.save('./label_all.npy', b)