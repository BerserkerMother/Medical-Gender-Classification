"""Dataset for Medical Research"""
import nibabel as nim
import glob
import os
import logging

import torch
from torch.utils.data import Dataset

from .path import DATA_PATH


class MedicalDataset(Dataset):
    def __init__(self, root: str, splits: str = 'train',
                 transform=None, ram: bool = False):
        """

        :param root: path to data folder
        :param split: dataset splits (train, val, test1, test2, test3)
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

        self.data = []
        self.images = []
        self.targets = []
        for split in self.splits:
            annotation_path = os.path.join(self.root, 'annotations', split + '.txt')
            with open(annotation_path, 'r') as file:
                text = file.read()[:-1]
            for line in text.split('\n'):
                line = line.split('\t')
                self.data.append((line[0], line[2]))
                self.targets.append(line[1])
                # transfer to ram if ram is True
                if self.ram:
                    image = nim.load(self.im_id2im_path[line[0]])
                    self.images.append(image)

        logging.info('SET %s Loaded\n# samples: %d' %
                     (splits, len(self.im_id2im_path)))

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

        # IO
        if self.ram:
            image = self.images[idx]
        else:
            # read nii image
            image = nim.load(self.im_id2im_path[im_id])
        image = torch.tensor(image.get_fdata(), dtype=torch.float)

        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)
        return image, age, target
