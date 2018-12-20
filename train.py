import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from database import ProteinDataset, batch_size, C, parallel
from models import TestDenseNet, FocalLoss, F1Loss, F1, acc, TestResNet
from utils import myOutput, saveOutput
import Augmentor


save_name = 'test16'
lossfunc = FocalLoss()
"""
# 1
start_epoches = 0  # >0 will resume your training
epoches = 2
rand_seed = 2
lr = 0.0001
net = TestDenseNet(finetune=False)
# 2
start_epoches = 2  # >0 will resume your training
epoches = 30
rand_seed = 230
lr = 0.0001
net = TestDenseNet(finetune=True)
"""
start_epoches = 18  # >0 will resume your training
epoches = 30
rand_seed = 1830
lr = 0.00001
net = TestDenseNet(finetune=True)

p = Augmentor.Pipeline()
p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.5)
# p.skew(probability=0.5, magnitude=0.5)
# p.random_erasing(probability=0.5, rectangle_area=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.75)
# p.skew(probability=0.5, magnitude=0.25)
# p.random_erasing(probability=0.5, rectangle_area=0.25)
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
                            transforms.Normalize(mean=[0.48, 0.48, 0.48, 0.48],
                                                 std=[0.22, 0.22, 0.22, 0.22])
                          ]))
validset = ProteinDataset(usezip=False,
                          mode='valid',
                          transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.48, 0.48, 0.48, 0.48],
                                                 std=[0.22, 0.22, 0.22, 0.22])
                          ]))
train_loader = DataLoader(trainset, batch_size=batch_size//4, shuffle=True, num_workers=16)
valid_loader = DataLoader(validset, batch_size=batch_size//4,  shuffle=True, num_workers=16)

# load net and loss
# print(net)
if parallel:
    net = torch.nn.DataParallel(net, device_ids=parallel)
if start_epoches:
    data = torch.load("{}_{}.pt".format(save_name, start_epoches))
    if parallel:
        net.module.load_state_dict(data)
    else:
        net.load_state_dict(data)

net.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)


# train
for epoch in range(start_epoches + 1, epoches + 1):
    train_pred = torch.zeros([len(trainset) * 4, C])
    train_targ = torch.zeros([len(trainset) * 4, C])
    train_loss = []

    net.train()
    for batch_idx, blob in enumerate(train_loader):
        optimizer.zero_grad()
        img = blob['img'].cuda()
        img = torch.cat([img[:, :, 0:224, 0:224],
                         img[:, :, 0:224, 224:448],
                         img[:, :, 224:448, 0:224],
                         img[:, :, 224:448, 224:448]])
        targ = blob['target'].cuda().repeat(4, 1)
        pred = net(img)
        img = []
        loss = lossfunc(pred, targ)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            now = batch_idx * batch_size
            train_loss.append(loss.cpu().numpy())
            train_pred[now: now + batch_size] = pred.cpu()
            train_targ[now: now + batch_size] = targ.cpu()
            myOutput(batch_idx, len(trainset) * 4, train_loss, train_pred, train_targ)

    valid_pred = torch.zeros([len(validset) * 4, C])
    valid_targ = torch.zeros([len(validset) * 4, C])
    valid_loss = []
    net.eval()
    with torch.no_grad():
        for batch_idx, blob in enumerate(valid_loader):
            img = blob['img'].cuda()
            img = torch.cat([img[:, :, 0:224, 0:224],
                             img[:, :, 0:224, 224:448],
                             img[:, :, 224:448, 0:224],
                             img[:, :, 224:448, 224:448]])
            targ = blob['target'].cuda().repeat(4, 1)
            pred = net(img)
            img = []
            valid_loss.append(lossfunc(pred, targ).cpu().numpy())
            now = batch_idx * batch_size
            valid_pred[now: now + batch_size] = pred
            valid_targ[now: now + batch_size] = targ
            myOutput(batch_idx, len(validset) * 4, valid_loss, valid_pred, valid_targ)

    print("epoch:{}/{}".format(epoch, epoches))
    print("Train: ", end='')
    myOutput(-1, len(trainset) * 4, train_loss, train_pred, train_targ)
    print("Valid: ", end='')
    myOutput(-1, len(validset) * 4, valid_loss, valid_pred, valid_targ)
    if parallel:
        torch.save(net.module.state_dict(), "{}_{}.pt".format(save_name, epoch))
    else:
        torch.save(net.state_dict(), "{}_{}.pt".format(save_name, epoch))
    saveOutput(save_name, epoch)
    print("---")
    
