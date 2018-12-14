import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from database import ProteinDataset, batch_size, C
from models import TestDenseNet, FocalLoss, F1Loss, F1, acc, TestResNet
from utils import myOutput
import Augmentor


start_epoches = 13  # >0 will resume your training
epoches = 20
rand_seed = 1220
save_name = 'test8'
lr = 0.0001
lossfunc = FocalLoss()
net = TestDenseNet(finetune=True)

p = Augmentor.Pipeline()
p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.5)
p.skew(probability=0.5, magnitude=0.5)
p.random_erasing(probability=0.5, rectangle_area=0.5)
# p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1)
# p.random_distortion(probability=0.5)

# init
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
history = []

# load data
trainset = ProteinDataset(usezip=False,
                          mode='train',
                          transform=transforms.Compose([
                            p.torch_transform(),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
                          ]))
validset = ProteinDataset(usezip=False,
                          mode='valid',
                          transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
                          ]))
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=2)

# load net and loss
if start_epoches:
    data = torch.load("{}_{}.pt".format(save_name, start_epoches))
    net.load_state_dict(data)

net.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)


# train
for epoch in range(start_epoches + 1, epoches + 1):
    train_pred = torch.zeros([len(trainset), C]).cuda()
    train_targ = torch.zeros([len(trainset), C]).cuda()
    train_loss = []

    net.train()
    for batch_idx, blob in enumerate(train_loader):
        optimizer.zero_grad()
        pred = net(blob['img'].cuda())
        targ = blob['target'].cuda()
        loss = lossfunc(pred, targ)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            now = batch_idx * batch_size
            train_loss.append(loss.cpu().numpy())
            train_pred[now: now + batch_size] = pred
            train_targ[now: now + batch_size] = targ
            myOutput(batch_idx, len(trainset), train_loss, train_pred, train_targ)

    valid_pred = torch.zeros([len(validset), C]).cuda()
    valid_targ = torch.zeros([len(validset), C]).cuda()
    valid_loss = []
    net.eval()
    with torch.no_grad():
        for batch_idx, blob in enumerate(valid_loader):
            pred = net(blob['img'].cuda())
            targ = blob['target'].cuda()
            valid_loss.append(lossfunc(valid_pred, valid_targ).cpu().numpy())
            now = batch_idx * batch_size
            valid_pred[now: now + batch_size] = pred
            valid_targ[now: now + batch_size] = targ
            myOutput(batch_idx, len(validset), valid_loss, valid_pred, valid_targ)

    print("epoch:{}/{}".format(epoch, epoches))
    print("Train: ", end='')
    myOutput(-1, len(trainset), train_loss, train_pred, train_targ)
    print("Valid: ", end='')
    myOutput(-1, len(validset), valid_loss, valid_pred, valid_targ)
    torch.save(net.state_dict(), "{}_{}.pt".format(save_name, epoch))
    print("---")
