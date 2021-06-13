import numpy as np
import Levenshtein
import config
from albumentations import (
    Compose, Normalize, Resize, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose
)
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def get_transforms(config, mode):
    if mode == 'train':
        return Compose([
            Resize(config.size, config.size),
            HorizontalFlip(p=0.5),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            #ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ])

    elif mode == 'valid':
        return Compose([
            Resize(config.size, config.size),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ])

def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    logit = pack_padded_sequence(logit, length, batch_first=True).data
    truth = pack_padded_sequence(truth, length, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=config.STOI['<pad>'])
    return loss
