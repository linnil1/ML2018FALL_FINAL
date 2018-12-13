import numpy as np
import torch
from database import batch_size
output_step = 100


def acc(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()


def f1(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return preds * 2 + targs


def macrof1(f1_score, verbose=0):
    if verbose:
        print('TP', (f1_score == 3).sum(dim=0))
        print('FP', (f1_score == 2).sum(dim=0))
        print('TN', (f1_score == 1).sum(dim=0))
        print('FN', (f1_score == 0).sum(dim=0))
    TP = (f1_score == 3).float() * 2
    other = TP + (f1_score == 2).float() + (f1_score == 1).float()

    score_f1 = (TP.sum(dim=0) / other.sum(dim=0)).cpu().numpy()
    if verbose:
        return score_f1
    else:
        return np.nanmean(score_f1)


def myOutput(batch_idx, length, ac, loss, f1):
    if batch_idx == -1 or batch_idx % output_step == 0:
        processbar(batch_idx, length)
        print("acc: {:.4f} loss: {:.4f} f1: {}".format(
              np.mean(ac), np.mean(loss), macrof1(torch.stack(f1))))


def processbar(batch_idx, length, end=' '):
    if batch_idx == -1:
        print("{:6}/{:6} 100%".format(length, length), end=end)
    else:
        print("{:6}/{:6} {:6.2f}%".format(
              batch_idx * batch_size, length,
              100 * batch_idx * batch_size / length), end=end)
