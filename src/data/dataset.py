"""Dataset for Medical Research"""
import nibabel as nim
import glob
import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

from .path import DATA_PATH


class MedicalDataset(Dataset):
    def __init__(self, root: str, splits: str = 'train',
                 transform=None, ram: bool = False, train=False):
        """

        :param root: path to data folder
        :param splits: dataset splits (train, val, test1, test2, test3)
        :param transform: 3D images transformations
        :param ram: if True then Images are copied to RAM for faster IO
        """
        self.root = root
        self.splits = splits.split("+")
        self.transform = transform
        self.ram = ram

        # get nii files path
        self.im_id2im_path = {}

        for split in self.splits:
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

        data = []
        targets = []
        for split in self.splits:
            annotation_path = os.path.join(self.root, 'annotations', split + '.txt')
            with open(annotation_path, 'r') as file:
                text = file.read()[:-1]
            for line in text.split('\n'):
                line = line.split('\t')
                # create extra information tuple
                extra = (float(line[2]), float(line[3]),
                         float(line[4]), float(line[8]),
                         float(line[9]), float(line[10]))
                data.append((line[0], extra))
                targets.append(line[1])

        if train:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            num_male = sum([1 for gen in targets if gen == 'M'])
            self.data = []
            self.targets = []
            num_female = 0
            for idx in indices:
                d = data[idx]
                t = targets[idx]
                if t == 'M':
                    self.data.append(d)
                    self.targets.append(t)
                elif num_female <= num_male:
                    self.data.append(d)
                    self.targets.append(t)
                    num_female += 1
        else:
            self.data = data
            self.targets = targets
        logging.info('SET %s Loaded\n# samples: %d' %
                     (splits, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """

        :param idx: data index
        :return: 3D MRI Image with Size (1, L, H, W), age group number, label
        """
        im_id, extra = self.data[idx]
        target = 0. if self.targets[idx] == 'M' else 1.
        age, TIV, GMv, GMn, WMn, CSFn = extra

        image = nim.load(self.im_id2im_path[im_id])
        image = torch.tensor(image.get_fdata(), dtype=torch.float)

        image = image.unsqueeze(0)  # (H, W, L) -> (1, H, W, L)
        if self.transform:
            image = self.transform(image)  # (C, H, W), (3, H, W), (H, W), (1, H, W)
        return image, age, TIV, GMv, GMn, WMn, CSFn, target, im_id

    def split_train_test(self, ratios):
        set_sizes = [int(r * len(self.data)) for r in ratios]
        test_size = len(self.data) - sum(set_sizes)
        set_sizes.append(test_size)

        return random_split(self, set_sizes)
