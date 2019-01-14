import numpy as np
import pandas as pd
import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


data_path = 'data'
# data_path = '/home/tmp'
train_csv = 'train.csv'
train_zip = 'train.zip'
test_csv = 'sample_submission.csv'
test_zip = 'test.zip'
train_folder = os.path.join(data_path, 'train')
test_folder = os.path.join(data_path, 'test')
colors = ['red', 'green', 'blue', 'yellow']
C = 28
num_train = 28000
np.set_printoptions(2)
parallel = [0, 1]
# parallel = [0]
crop_num = 5
batch_size = 5 * crop_num * len(parallel)


dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48, 0.48, 0.48, 0.48],
                         std=[0.22, 0.22, 0.22, 0.22])
])


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
    def __init__(self, mode='train', transform=None, usezip=True, num_train=num_train):
        if mode == 'test':
            self.groundtrue = pd.read_csv(os.path.join(data_path, test_csv))
        else:
            self.groundtrue = pd.read_csv(os.path.join(data_path, train_csv))
            if len(self.groundtrue) < num_train:
                raise ValueError("number of data is " + str(len(self.groundtrue)))

        self.transform = transform
        self.usezip = usezip
        self.mode = mode
        self.num_train = num_train
        if usezip:
            if mode == 'test':
                self.myzip = zipfile.ZipFile(os.path.join(data_path, test_zip))
            else:
                self.myzip = zipfile.ZipFile(os.path.join(data_path, train_zip))

    def __len__(self):
        if self.mode == 'train':
            return self.num_train
        elif self.mode == 'valid':
            return len(self.groundtrue) - self.num_train
        else:
            return len(self.groundtrue)

    def __getitem__(self, idx):
        if self.mode == 'valid':
            idx += self.num_train
        # validation is at the begining
        # if self.mode == 'train':
        #     idx += len(self.groundtrue) - self.num_train
        name, target = self.groundtrue.iloc[idx]
        imagefile = [name + '_' + color + '.png' for color in colors]
        if self.usezip:
            imagefile = [self.myzip.open(f) for f in imagefile]
        elif self.mode == 'test':
            imagefile = [os.path.join(test_folder, f) for f in imagefile]
        else:
            imagefile = [os.path.join(train_folder, f) for f in imagefile]
        img = np.stack([np.array(Image.open(f)) for f in imagefile])
        img = Image.fromarray(np.rollaxis(img, 0, 3))

        if self.mode != 'test':
            one_hot = np.zeros(C).astype(np.float32)
            one_hot[list(map(int, target.split()))] = 1

        if self.transform:
            img = self.transform(img)

        if self.mode == 'test':
            return {'img': img, 'name': name}
        else:
            return {'img': img, 'target': one_hot}
