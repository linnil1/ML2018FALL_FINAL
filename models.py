from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet121
from collections import OrderedDict


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
    """ https://pytorch.org/docs/stable/_modules/torchvision/models/densenet.html#densenet201 """

    def __init__(self, finetune=True):
        super(TestDenseNet, self).__init__()
        self.dense = densenet121(pretrained=True)

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('dense', self.dense.features[1:]),
        ]))

        self.linear = nn.Sequential(OrderedDict([
            ('classifier', self.dense.classifier),
            ('linear', nn.Linear(1000, 28))
        ]))

        # self.features[0].weight[:, :3, :, :] = self.dense.features[0].weight[:]

        if not finetune:
            for param in self.features[1].parameters():
                param.requires_grad = False
            for param in self.linear[0].parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.linear(out)
        return out


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
