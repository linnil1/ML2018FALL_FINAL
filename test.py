import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from database import ProteinDataset
from models import TestDenseNet, TestNet, FocalLoss, acc


save_name = 'test_0.pt'
net = TestDenseNet()
net.cuda()

data = torch.load(save_name)
net.load_state_dict(data)
net.eval()
net.cuda()

trainset = ProteinDataset(usezip=False,
                          transform=transforms.Compose([
                            # transforms.RandomVerticalFlip(),
                            # transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                          ]))
data_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=6)

arr_ac = []
for batch_idx, blob in enumerate(data_loader):
    pred = net(blob['img'].cuda())
    target = blob['target'].cuda()
    ac = acc(pred, target).cpu().numpy()
    arr_ac.append(ac)
    print("{}/{} {:.2f}%".format(batch_idx, len(trainset),
                                 100 * batch_idx / len(trainset)), ac)

print(np.array(arr_ac).mean())
