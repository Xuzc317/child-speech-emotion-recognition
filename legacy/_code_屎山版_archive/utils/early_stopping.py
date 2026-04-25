# utils/early_stopping.py
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        """
        Args:
            patience (int): 容忍多少个epoch没有提升
            delta (float): 认为提升的最小变化量
            verbose (bool): 是否打印提示信息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0