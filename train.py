import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121
import numpy as np
import matplotlib.pyplot as plt

from database import ProteinDataset
from models import TestDenseNet, FocalLoss, acc

epoches = 2
rand_seed = 12345
save_name = 'test'
lr = 0.001

# ------------
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


trainset = ProteinDataset(usezip=False,
                          transform=transforms.Compose([
                           # transforms.RandomVerticalFlip(),
                           # transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                          ]))
data_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=6)


net = TestDenseNet()
net.cuda()


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
lossfunc = FocalLoss()
history_loss = []
history_acc = []

for epoch in range(epoches):
    for batch_idx, blob in enumerate(data_loader):
        optimizer.zero_grad()
        pred = net(blob['img'].cuda())
        target = blob['target'].cuda()
        loss = lossfunc(pred, target)
        loss.backward()
        optimizer.step()

        ac = acc(pred, target).cpu().numpy()
        print("{}/{} {:.2f}%".format(batch_idx, len(trainset),
                                     100 * batch_idx / len(trainset)), ac)
        history_loss.append(loss)
        history_acc.append(ac)
    torch.save(net.state_dict(), "{}_{}.pt".format(save_name, epoch))

plt.plot(history_loss)
plt.plot(history_acc)
plt.show()
