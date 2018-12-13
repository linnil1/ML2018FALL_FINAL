import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from database import ProteinDataset
from models import TestDenseNet, TestNet, FocalLoss
import pandas as pd


save_name = 'test_11.pt'
output_name = 'test.csv'
batch_size = 8
output_step = 100

# init network
net = TestDenseNet()
data = torch.load(save_name)
net.load_state_dict(data)
net.eval()
net.cuda()

testset = ProteinDataset(usezip=False,
                         mode='test',
                         transform=transforms.Compose([
                           # transforms.RandomVerticalFlip(),
                           # transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                         ]))

test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=6)

ans_dict = {}

for batch_idx, blob in enumerate(test_loader):
    if batch_idx % output_step == 0:
        print("{}/{} {:.2f}%".format(
              batch_idx * batch_size, len(testset),
              100 * batch_idx * batch_size / len(testset)))
    # run
    pred = net(blob['img'].cuda()) >= 0
    name = blob['name']
    for i in range(len(blob['img'])):
        pred_ind = pred[i].nonzero().cpu().numpy()
        ans = ''
        if pred_ind.any():
            ans = ' '.join(str(p) for p in pred_ind[:, 0])
        ans_dict[name[i]] = ans

# write csv
ans = []
for index, row in testset.groundtrue.iterrows():
    ans.append([row[0], ans_dict[row[0]]])
df = pd.DataFrame(ans, columns=['Id', 'Predicted'])
df.to_csv(output_name, index=False)
