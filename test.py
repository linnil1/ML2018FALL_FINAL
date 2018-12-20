import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from database import ProteinDataset, batch_size
from models import TestDenseNet, FocalLoss, evalute
import pandas as pd
from utils import processbar


# kaggle competitions submit -c human-protein-atlas-image-classification -f submission.csv -m "Message" 
save_name = 'test16_29.pt'
output_name = 'test16_29_max.csv'
nozero = True

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
                           # transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
                           transforms.Normalize(mean=[0.48, 0.48, 0.48, 0.48],
                                                std=[0.22, 0.22, 0.22, 0.22])
                         ]))

test_loader = DataLoader(testset, batch_size=batch_size//4, shuffle=False, num_workers=8)

ans_dict = {}

for batch_idx, blob in enumerate(test_loader):
    processbar(batch_idx + 1, len(testset) * 4, end='\n')

    # run
    img = blob['img'].cuda()
    real_len = len(img)
    img = torch.cat([img[:, :, 0:224, 0:224],
                     img[:, :, 0:224, 224:448],
                     img[:, :, 224:448, 0:224],
                     img[:, :, 224:448, 224:448]])
    res = net(img)
    for i in range(real_len):
        a = res[i::real_len]
        assert(len(a) == 4)
        # res[i] = a.mean(dim=0)
        res[i] = a.max(dim=0)[0]
        a = None
    res = res[:real_len]
    pred = evalute(res)
    name = blob['name']
    for i in range(len(blob['img'])):
        pred_ind = pred[i].nonzero().cpu().numpy()
        ans = ''
        if pred_ind.any():
            ans = ' '.join(str(p) for p in pred_ind[:, 0])
        elif nozero:
            ans = str(torch.argmax(res[i]).cpu().numpy())
        # print(ans)
        ans_dict[name[i]] = ans
    img = None
    res = None
    pred = None

# write csv
ans = []
for index, row in testset.groundtrue.iterrows():
    ans.append([row[0], ans_dict[row[0]]])
df = pd.DataFrame(ans, columns=['Id', 'Predicted'])
df.to_csv(output_name, index=False)
