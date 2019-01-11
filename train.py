import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time

from database import ProteinDataset, batch_size, C, parallel, dataset_transform, crop_num
from models import TestDenseNet, FocalLoss, F1Loss, F1, acc, TestResNet, F1Focal
from utils import myOutput, saveOutput
import Augmentor


save_name = 'test36'
lossfunc = F1Focal()
"""
# test36
# densenet 1
start_epoches = 0  # >0 will resume your training
epoches = 20
rand_seed = 201
lr = 0.0001
net = TestDenseNet(finetune=False)
# densenet 2
start_epoches = 19  # >0 will resume your training
epoches = 50
rand_seed = 1950
lr = 0.00004
net = TestDenseNet(finetune=True)
"""
# densenet 1
start_epoches = 0  # >0 will resume your training
epoches = 20
rand_seed = 201
lr = 0.0001
net = TestDenseNet(finetune=False)

p = Augmentor.Pipeline()
# p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
# p.skew(probability=0.5, magnitude=0.5)
# p.zoom_random(probability=0.25, percentage_area=0.75)
# p.skew(probability=0.25, magnitude=0.25)
# p.random_erasing(probability=0.5, rectangle_area=0.25)
# p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1)

# init
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
history = []
pt_name = save_name + "_{:02}.pt"
print(save_name)


trainset_transform = transform=transforms.Compose([
    p.torch_transform(),
    dataset_transform])

# load data
trainset = ProteinDataset(usezip=False,
                          mode='train',
                          transform=transforms.Compose([
                            transforms.FiveCrop(256),
                            transforms.Lambda(lambda crops: torch.stack(
                                [trainset_transform(crop) for crop in crops]))
                          ]))
validset = ProteinDataset(usezip=False,
                          mode='valid',
                          transform=transforms.Compose([
                            transforms.FiveCrop(256),
                            transforms.Lambda(lambda crops: torch.stack(
                                [dataset_transform(crop) for crop in crops]))
                          ]))

train_loader = DataLoader(trainset, batch_size=batch_size // crop_num, shuffle=True, num_workers=16)
valid_loader = DataLoader(validset, batch_size=batch_size // crop_num, shuffle=True, num_workers=16)

# load net and loss
# print(net)
if parallel:
    net = torch.nn.DataParallel(net, device_ids=parallel)
if start_epoches:
    data = torch.load(pt_name.format(start_epoches))
    if parallel:
        net.module.load_state_dict(data)
    else:
        net.load_state_dict(data)

net.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)


# train
for epoch in range(start_epoches + 1, epoches + 1):
    t = time.time()
    train_pred = torch.zeros([len(trainset) * crop_num, C])
    train_targ = torch.zeros([len(trainset) * crop_num, C])
    train_loss = []

    net.train()
    for batch_idx, blob in enumerate(train_loader):
        optimizer.zero_grad()
        img = blob['img'].cuda()
        img = img.view(-1, *img.size()[2:])
        # shuf = torch.randperm(len(img))
        shuf = torch.arange(len(img)).view(-1, crop_num).t().contiguous().view(-1)
        targ = blob['target'].cuda().repeat(1, crop_num).view(-1, C)[shuf]
        pred = net(img[shuf])
        img = []
        loss = lossfunc(pred, targ)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            now = batch_idx * batch_size
            train_loss.append(loss.cpu().numpy())
            train_pred[now: now + batch_size] = pred.cpu()
            train_targ[now: now + batch_size] = targ.cpu()
            myOutput(batch_idx, len(train_pred), train_loss, train_pred, train_targ)

    valid_pred = torch.zeros([len(validset) * crop_num, C])
    valid_targ = torch.zeros([len(validset) * crop_num, C])
    valid_loss = []
    net.eval()
    with torch.no_grad():
        for batch_idx, blob in enumerate(valid_loader):
            img = blob['img'].cuda()
            img = img.view(-1, *img.size()[2:])
            # shuf = torch.randperm(len(img))
            shuf = torch.arange(len(img)).view(-1, crop_num).t().contiguous().view(-1)
            targ = blob['target'].cuda().repeat(1, crop_num).view(-1, C)[shuf]
            pred = net(img[shuf])
            img = []
            valid_loss.append(lossfunc(pred, targ).cpu().numpy())
            now = batch_idx * batch_size
            valid_pred[now: now + batch_size] = pred
            valid_targ[now: now + batch_size] = targ
            myOutput(batch_idx, len(valid_pred), valid_loss, valid_pred, valid_targ)

    print("epoch:{}/{}".format(epoch, epoches))
    print("Train: ", end='')
    myOutput(-1, len(train_pred), train_loss, train_pred, train_targ)
    print("Valid: ", end='')
    myOutput(-1, len(valid_pred), valid_loss, valid_pred, valid_targ)
    if parallel:
        torch.save(net.module.state_dict(), pt_name.format(epoch))
    else:
        torch.save(net.state_dict(), pt_name.format(epoch))
    saveOutput(save_name, epoch)
    print("Time:", int(time.time() - t), '(s)')
    print("---")
    
