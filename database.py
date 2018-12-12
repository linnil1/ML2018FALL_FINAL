import numpy as np
import pandas as pd
import os
import zipfile 
from PIL import Image
from torch.utils.data import Dataset


data_path = 'data'
train_csv = 'train.csv'
train_zip = 'train.zip'
train_folder = os.path.join(data_path, 'train')
# colors = ['red', 'green', 'blue', 'yellow']
colors = ['red', 'green', 'blue']
C = 28

def test():
    df = pd.read_csv(os.path.join(data_path, train_csv))
    name = df.iloc[0, 0]
    print(name)

    myzip = zipfile.ZipFile(os.path.join(data_path, train_zip))
    # myzip.infolist()
    import matplotlib.pyplot as plt
    plt.imshow(plt.imread(myzip.open(name + '_blue.png')))
    plt.show()


class ProteinDataset(Dataset):
    """ Human Protein Atlas Image """
    def __init__(self, transform=None, usezip=True):
        self.groundtrue = pd.read_csv(os.path.join(data_path, train_csv))
        self.transform = transform
        self.usezip = usezip
        if usezip:
            self.myzip = zipfile.ZipFile(os.path.join(data_path, train_zip))

    def __len__(self):
        return len(self.groundtrue)

    def __getitem__(self, idx):
        # print(idx)
        name, target = self.groundtrue.iloc[idx]
        # print([self.myzip.getinfo(name + '_' + color + '.png') for color in colors])
        imagefile = [name + '_' + color + '.png' for color in colors]
        if self.usezip:
            imagefile = [self.myzip.open(f) for f in imagefile]
        else:
            imagefile = [os.path.join(train_folder, f) for f in imagefile]
        img = np.stack([np.array(Image.open(f)) for f in imagefile])
        img = Image.fromarray(np.rollaxis(img, 0,3))
        img = img.resize([224, 224])

        one_hot = np.zeros(C).astype(np.float32)
        one_hot[list(map(int, target.split()))] = 1

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'target': one_hot}
