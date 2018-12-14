import numpy as np
import torch
from database import batch_size
from models import F1, acc
output_step = 50


def myOutput(batch_idx, length, loss,  pred, targ):
    batch_idx += 1
    if batch_idx % output_step == 0:
        now = batch_idx * batch_size
        if not batch_idx:
            now = length
            batch_idx = length
        processbar(batch_idx, length)
        ac = acc(pred[:now], targ[:now]).cpu().numpy()
        f1 = F1((pred).float(), targ).detach().cpu().numpy()
        print("acc: {:.4f} loss: {:.4f} f1: {}".format(
              ac, np.mean(loss), f1))


def processbar(batch_idx, length, end=' '):
    if batch_idx * batch_size < length:
        print("{:6}/{:6} {:6.2f}%".format(
              batch_idx * batch_size, length,
              100 * batch_idx * batch_size / length), end=end)
    else:
        print("{:6}/{:6} 100%".format(length, length), end=end)
