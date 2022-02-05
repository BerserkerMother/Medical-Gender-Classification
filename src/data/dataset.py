"""Dataset for Medical Research"""
import nibabel as nim
import glob
import os
import logging

import torch
from torch.utils.data import Dataset

from .path import DATA_PATH


class MedicalDataset(Dataset):
    def __init__(self, root: str, split: str = 'train',
                 transform=None):
        """

        :param root: path to data folder
        :param split: dataset splits (train, val, test1, test2, test3)
        """
        self.root = root
        self.split = split
        self.transform = transform

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
            line = line.split('\t')
            self.data.append((line[0], line[2]))
            self.targets.append(line[1])

        logging.info('SET %s Loaded\n# samples: %d' %
                     (split, len(self.im_id2im_path)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """

        :param idx: data index
        :return: 3D MRI Image with Size (1, L, H, W), age group number, label
        """
        im_id, age = self.data[idx]
        target = 0. if self.targets[idx] == 'M' else 1.

        age = int((float(age) - 20) / 5)
        # load nii image
        image = nim.load(self.im_id2im_path[im_id])
        image = torch.tensor(image.get_fdata(), dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)
        return image, age, target
