import numpy as np
from database import ProteinDataset
import pandas as pd

ens_model = ['test30_22.npz', 'test34_30.npz']
result_file = 'test30_22,test34_30,mean.csv'

data = [np.load(f) for f in ens_model]

datadict = [dict(zip(d['name'], d['pred'])) for d in data]

testset = ProteinDataset(usezip=False, mode='test')

ans = []
for index, row in testset.groundtrue.iterrows():
    pred = np.vstack([d[row[0]] for d in datadict])
    # ens_pred = pred.max(axis=0)
    ens_pred = pred.mean(axis=0)
    ens_pred_ind = np.nonzero(ens_pred >= 0.5)[0]
    if any(ens_pred_ind):
        s = ' '.join(str(p) for p in ens_pred_ind)
    else:
        s = str(np.argmax(ens_pred))
    ans.append([row[0], s])

# write to csv
df = pd.DataFrame(ans, columns=['Id', 'Predicted'])
df.to_csv(result_file, index=False)
