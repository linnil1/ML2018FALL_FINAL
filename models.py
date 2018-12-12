from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet121


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(3, 10, 5, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear = nn.Linear(2601000, 28)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class TestDenseNet(nn.Module):
    def __init__(self):
        super(TestDenseNet, self).__init__()
        self.dense = densenet121(pretrained=True)
        self.linear = nn.Linear(1000, 28)
        for param in self.dense.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.dense(x)
        x = self.linear(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.sum(dim=1).mean()


def acc(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()
