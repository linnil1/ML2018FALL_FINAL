import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121
import numpy as np
import matplotlib.pyplot as plt

from database import ProteinDataset, batch_size
from models import TestDenseNet, FocalLoss
from utils import acc, f1, macrof1, myOutput
import Augmentor

start_epoches = 0  # >0 will resume your training
epoches = 12
rand_seed = 12345
save_name = 'test2'
lr = 0.001

# init
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
history = []

p = Augmentor.Pipeline()
p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1)
# p.random_distortion(probability=0.5)
p.random_erasing(probability=0.5, rectangle_area=0.4)

# load data
trainset = ProteinDataset(usezip=False,
                          mode='train',
                          transform=transforms.Compose([
                            p.torch_transform(),
                            transforms.ToTensor(),
                          ]))
validset = ProteinDataset(usezip=False,
                          mode='valid',
                          transform=transforms.Compose([
                            transforms.ToTensor(),
                          ]))
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=4)

# load net and loss
net = TestDenseNet(finetune=True)
if start_epoches:
    data = torch.load("{}_{}.pt".format(save_name, start_epoches))
    net.load_state_dict(data)

net.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
lossfunc = FocalLoss()


# train
for epoch in range(start_epoches + 1, epoches):
    train_acc = []
    train_loss = []
    train_f1 = []
    for batch_idx, blob in enumerate(train_loader):
        optimizer.zero_grad()
        pred = net(blob['img'].cuda())
        target = blob['target'].cuda()
        loss = lossfunc(pred, target)
        loss.backward()
        optimizer.step()

        ac = acc(pred, target).cpu().numpy()
        train_acc.append(ac)
        f1_score = f1(pred, target)
        train_f1.extend(f1_score)
        train_loss.append(loss.detach().cpu().numpy())

        myOutput(batch_idx, len(trainset), train_acc, train_loss, train_f1)

    valid_acc = []
    valid_loss = []
    valid_f1 = []
    with torch.no_grad():
        for batch_idx, blob in enumerate(valid_loader):
            pred = net(blob['img'].cuda())
            target = blob['target'].cuda()

            ac = acc(pred, target).cpu().numpy()
            valid_acc.append(ac)
            f1_score = f1(pred, target)
            valid_f1.extend(f1_score)
            loss = lossfunc(pred, target)
            valid_loss.append(loss.cpu().numpy())

            myOutput(batch_idx, len(validset), valid_acc, valid_loss, valid_f1)

    print("epoch:{}/{}".format(epoch, epoches))
    print("Train: ", end='')
    myOutput(-1, len(trainset), train_acc, train_loss, train_f1)
    print("Valid: ", end='')
    myOutput(-1, len(validset), valid_acc, valid_loss, valid_f1)
    torch.save(net.state_dict(), "{}_{}.pt".format(save_name, epoch))
    print("---")

plt.subplot(121)
plt.plot(np.arange(start_epoches + 1, epoches), train_acc.mean(axis=1), label='train')
plt.plot(np.arange(start_epoches + 1, epoches), valid_acc.mean(axis=1), label='valid')
plt.legend()
plt.subplot(122)
plt.plot(np.arange(start_epoches + 1, epoches), train_loss.mean(axis=1), label='train')
plt.plot(np.arange(start_epoches + 1, epoches), valid_loss.mean(axis=1), label='valid')
plt.show()
