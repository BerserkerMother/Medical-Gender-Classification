"""Dataset for Medical Research"""
import torch
from torch.utils.data import Dataset

import nibabel as nim
import glob
import os

from .path import DATA_PATH


class MedicalDataset(Dataset):
    def __init__(self, root: str, split: str = 'train'):
        """

        :param root: path to data folder
        :param split: dataset splits (train, val, test1, test2, test3)
        """
        self.root = root
        self.split = split

        # get nii files path
        self.im_id2im_path = {}
        if split == 'train':
            feature_path = os.path.join(root, DATA_PATH[split])
            for path in glob.glob(feature_path + '/*'):
                ID = path.split('/')[-1].split('.')[0][5:]
                self.im_id2im_path[ID] = path
        else:
            feature_path = os.path.join(root, DATA_PATH[split])
            for path in glob.glob(feature_path + '/*'):
                ID = path.split('/')[-1].split('.')[0][5:14]
                self.im_id2im_path[ID] = path

        annotation_path = os.path.join(self.root, 'annotations', split + '.txt')
        self.data = []
        self.targets = []
        with open(annotation_path, 'r') as file:
            text = file.read()[:-1]
        for line in text.split('\n'):
            line = line.split(' ')
            self.data.append((line[0], line[2]))
            self.targets.append(line[1])

        print('SET %s Loaded\n# samples: %d' % (split, len(self.im_id2im_path)))
        print('_' * 100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_id, age = self.data[idx]
        target = 0. if self.targets[idx] == 'M' else 1.

        age = int((float(age) - 20) / 5)
        # load nii image
        image = nim.load(self.im_id2im_path[im_id])
        image = torch.tensor(image.get_fdata(), dtype=torch.float)
        image = image.unsqueeze(0)

        return image, age, target
