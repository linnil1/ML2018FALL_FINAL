import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from database import ProteinDataset, batch_size, dataset_transform, crop_num
from models import TestDenseNet, FocalLoss, evalute, TestResNet
import pandas as pd
from utils import processbar


# kaggle competitions submit -c human-protein-atlas-image-classification -f submission.csv -m "Message" 
save_name = 'test30_22.pt'
output_name = 'test30_22.csv'
nozero = True

# init network
# net = TestResNet()
net = TestDenseNet()
data = torch.load(save_name)
net.load_state_dict(data)
net.eval()
net.cuda()

testset = ProteinDataset(usezip=False,
                         mode='test',
                         transform=transforms.Compose([
                           transforms.FiveCrop(256),
                           transforms.Lambda(lambda crops: torch.stack(
                               [dataset_transform(crop) for crop in crops]))
                         ]))

test_loader = DataLoader(testset, batch_size=batch_size // crop_num, shuffle=False, num_workers=8)

ans_dict = {}

for batch_idx, blob in enumerate(test_loader):
    processbar(batch_idx + 1, len(testset) * crop_num, end='\n')

    # run
    img = blob['img'].cuda()
    real_len = len(img)
    img = img.view(-1, *img.size()[2:])
    res = net(img)
    img = None
    for i in range(real_len):
        a = res[i * crop_num:i * crop_num + crop_num]
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
        print(ans)
        ans_dict[name[i]] = ans
    res = None
    pred = None

# write csv
ans = []
for index, row in testset.groundtrue.iterrows():
    ans.append([row[0], ans_dict[row[0]]])
df = pd.DataFrame(ans, columns=['Id', 'Predicted'])
df.to_csv(output_name, index=False)
