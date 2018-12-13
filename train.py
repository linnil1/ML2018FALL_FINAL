import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121
import numpy as np
import matplotlib.pyplot as plt

from database import ProteinDataset
from models import TestDenseNet, FocalLoss
from utils import acc, f1, macrof1

epoches = 12
rand_seed = 12345
save_name = 'test'
output_step = 100
lr = 0.001
batch_size = 8

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
                            # transforms.RandomVerticalFlip(),
                            # transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                          ]))
validset = ProteinDataset(usezip=False,
                          mode='valid',
                          transform=transforms.Compose([
                            # transforms.RandomVerticalFlip(),
                            # transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                          ]))
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=6)

# load net and loss
net = TestDenseNet()
net.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
lossfunc = FocalLoss()


def myOutput(batch_idx, length, ac):
    if batch_idx % output_step == 0:
        print("{}/{} {:.2f}% acc: {:.2f}".format(
              batch_idx * batch_size, length,
              100 * batch_idx * batch_size / length,
              ac))


# train
for epoch in range(epoches):
    train_acc = []
    train_f1 = []
    for batch_idx, blob in enumerate(train_loader):
        if batch_idx > 100:
            break
        optimizer.zero_grad()
        pred = net(blob['img'].cuda())
        target = blob['target'].cuda()
        loss = lossfunc(pred, target)
        loss.backward()
        optimizer.step()

        ac = acc(pred, target).cpu().numpy()
        train_acc.append(ac)
        f1_score = f1(pred, target)
        train_f1.append(f1_score)

        myOutput(batch_idx, len(trainset), ac)

    valid_acc = []
    valid_f1 = []
    with torch.no_grad():
        for batch_idx, blob in enumerate(valid_loader):
            pred = net(blob['img'].cuda())
            target = blob['target'].cuda()

        ac = acc(pred, target).cpu().numpy()
        valid_acc.append(ac)
        f1_score = f1(pred, target)
        valid_f1.append(f1_score)

        myOutput(batch_idx, len(validset), ac)

        train_f1 = macrof1(torch.stack(train_f1))
        valid_f1 = macrof1(torch.stack(valid_f1))

    print("epoch:{}/{}".format(epoch, epoches))
    print("train:", np.mean(train_acc))
    print("valid:", np.mean(valid_acc))
    print("train f1:", np.mean(train_f1))
    print("valid f1:", np.mean(valid_f1))
    torch.save(net.state_dict(), "{}_{}.pt".format(save_name, epoch))

plt.plot(history_loss)
plt.plot(history_acc)
plt.show()
