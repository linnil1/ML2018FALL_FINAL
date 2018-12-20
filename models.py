import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet121, densenet201, resnet101
from collections import OrderedDict


class TestDenseNet(nn.Module):
    """ https://pytorch.org/docs/stable/_modules/torchvision/models/densenet.html"""

    def __init__(self, finetune=True):
        super(TestDenseNet, self).__init__()
        # self.dense = densenet121(pretrained=True)
        self.dense = densenet201(pretrained=True)

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            # dense161
            # ('conv0', nn.Conv2d(4, 96, kernel_size=7, stride=2, padding=3, bias=False)),
            ('dense', self.dense.features[1:]),
            ('dense-relu', nn.ReLU(inplace=True)),
        ]))

        self.linear = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(1920 * 4 * 4, 1000)),
            # ('linear0', nn.Linear(16384, 1000)), # for dense121
            # ('relu0', nn.ReLU()),
            # ('dropout0', nn.Dropout(0.2)),
            ('linear1', nn.Linear(1000, 28)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        # self.adapt_maxpool = nn.AdaptiveMaxPool2d([2, 2], return_indices=False)
        # self.adapt_maxpool = nn.AdaptiveAvgPool2d([2, 2])

        # self.features[0].weight[:, :3, :, :] = self.dense.features[0].weight[:]

        if not finetune:
            for param in self.features[1].parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=3)
        # x = self.adapt_maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def evalute(x):
    return (x >= 0.5).int()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.eps = 1e-10

    def forward(self, pred, targ):
        x = torch.zeros(targ.size()).cuda()
        x[targ == 1] = pred[targ == 1]
        x[targ == 0] = 1 - pred[targ == 0]
        x[x < self.eps] += self.eps
        return -((1 - x).pow(self.gamma) * x.log()).sum(dim=1).mean()
        """
        max_val = (-input).clamp(min=0)
        loss = input - input * targ + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (targ * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.sum(dim=1).mean()
        """


class F1Loss(nn.Module):
    def forward(self, pred, targ):
        tp = (pred * targ).sum(dim=0)
        fp = (pred * (1 - targ)).sum(dim=0)
        tn = ((1 - pred) * targ).sum(dim=0)

        f1 = 2 * tp / (2 * tp + fp + tn)
        return 1 - (f1[torch.isfinite(f1)]).mean()


def F1(pred, targ):
    predint = evalute(pred)
    targint = targ.int()
    tp = ((predint == 1) & (targint == 1)).sum(dim=0).float()
    fp = ((predint == 0) & (targint == 1)).sum(dim=0).float()
    tn = ((predint == 1) & (targint == 0)).sum(dim=0).float()

    f1 = 2 * tp / (2 * tp + fp + tn)
    return (f1[torch.isfinite(f1)]).mean()


def acc(pred, targ):
    return (evalute(pred) == targ.int()).float().mean()


class TestResNet(nn.Module):
    """ https://pytorch.org/docs/stable/_modules/torchvision/models/inception.html """

    def __init__(self, finetune=True):
        super(TestResNet, self).__init__()
        self.resnet = resnet101(pretrained=True)

        oldweight = self.resnet.conv1.weight
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # self.resnet.conv1.weight[:, :3] = oldweight

        self.linear = nn.Sequential(OrderedDict([
            # ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('linear', nn.Linear(1000, 28)),
        ]))

        if not finetune:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc.requires_grad = True
            self.resnet.conv1.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        x = self.linear(x)
        return x
