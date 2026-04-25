import os
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from transformers import  Wav2Vec2Model

class EmotionDataset(Dataset):
    def __init__(self, wav_dir, label_file, transform=None):
        """
        Args:
            wav_dir (str): Path to the directory with WAV files.
            label_file (str): Path to the label file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.wav_dir = wav_dir
        self.transform = transform

        self.labels = self._parse_labels(label_file)
        self.file_list = list(self.labels.keys())
        # self.file_list = self.deal_label(label_file)
        self.w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def deal_label(self, label_file):
        file_list = []
        with open(label_file, 'r') as f:
            for line in f:
                file_list.append([line.strip().split()[1], line.strip().split()[0]])
        return file_list

    def _parse_labels(self, label_file):
        labels = defaultdict(list)
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                labels[parts[1]].append(self._label_to_index(parts[0])) # without N
                # labels[parts[0]].append(self._label_to_index(parts[1])) # with N
        return labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        wav_path = os.path.join(self.wav_dir, f"{file_id}.wav")
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.to('cuda')
        
        wav2vec2 = self.w2v(waveform)['last_hidden_state']
        
        label = self.labels[file_id]
        # Convert label to index (assuming you have a mapping from label to index)
        # label_index = self._label_to_index(label)
        
        return wav2vec2, label

    def get_valid_indices(self):
        valid_indices = []
        for idx, file_id in enumerate(self.file_list):
            if self.labels[file_id] != 'X':
                valid_indices.append(idx)
        return valid_indices

    def _label_to_index(self, label):
        # Assuming you have a mapping from label to index
        label_mapping = {
            'J': 0, 'S': 1, 'M': 2, 'N': 3, 'O': 4,
            'B': 5, 'E': 6, 'H': 7, 'T': 8, 'R': 9,
            'A': 10, 'X': 11  # X represents no label
        }
        label_mapping = {
            'N': 0, 'R': 1, 'E': 2, 'A': 3, 'P': 4 
        }
        return label_mapping[label]

def collate_fn(batch):
    # batch 是一个列表，包含 (wav, label) 元组
    # 提取 wav 和 label
    wavs = [item[0].squeeze(0) for item in batch]  # 去掉 batch 维度
    labels = [torch.tensor(item[1][0]) for item in batch]

    # 按长度降序排序（pack_sequence 的要求）
    wavs.sort(key=lambda x: x.size(0), reverse=True)

    # 使用 pack_sequence 打包 wavs
    # wavs = [seq.unsqueeze(-1) for seq in wavs]
    padded_wavs = pad_sequence(wavs, batch_first=False)

    lengths = [wav.size(0) for wav in wavs]
    mask = torch.zeros((padded_wavs.size(0), padded_wavs.size(1)), dtype=torch.bool)  # 形状为 [max_len, batch_size]
    for i, length in enumerate(lengths):
        mask[:length, i] = True  # 标记有效位置为 True
    # 将 labels 转换为张量
    labels = torch.stack(labels)

    # 返回打包后的 wavs 和 labels
    return padded_wavs, labels, mask


if __name__ == "__main__":
    # Example usage:
    wav_dir = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\wav'
    # label_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\word_labels_11cl_corpus.txt'
    label_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\word_labels_5cl_corpus_out.txt'
    dataset = EmotionDataset(wav_dir, label_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for wavs, labels in dataloader:
        wav = wavs
        label = labels
        exit()