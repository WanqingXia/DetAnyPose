import glob
import hashlib
import logging
import os
import random
import shutil

from pathlib import Path
import scipy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def create_dataloader(path, type, imgsz, batch_size, workers=8):
    dataset = LoadImagesAndLabels(path, type, imgsz)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             sampler=torch.utils.data.RandomSampler(dataset),
                                             pin_memory=True,
                                             )
    return dataloader, dataset


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, type, img_size=256):
        self.img_size = img_size
        assert type in ('train', 'test'), f'{type} is not train or test'
        self.type = type
        self.data_paths = []
        if self.type == 'train':
            self.path = Path(path) / 'YCB_pairs' / 'train_data'
        elif self.type == 'test':
            self.path = Path(path) / 'YCB_pairs' / 'test_data'

        try:
            self.data_paths = os.listdir(self.path)
            print('{} training sets are found'.format(self.__len__()))
        except:
            print('Cache does not exist in {}, please run utils/process_data.py first.'.format(self.path))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.path / self.data_paths[index]

        # Convert
        color_anc = self.load_color(data_path / 'color_original.png')
        color_pos = self.load_color(data_path / 'color_generated.png')
        color_neg = self.load_color(data_path / 'color_negative.png')

        depth_anc = self.load_depth(data_path / 'depth_original.png')
        depth_pos = self.load_depth(data_path / 'depth_generated.png')
        depth_neg = self.load_depth(data_path / 'depth_negative.png')

        pose_anc = torch.from_numpy(np.loadtxt(data_path / 'pose_original.txt'))
        pose_pos = torch.from_numpy(np.loadtxt(data_path / 'pose_generated.txt'))

        with open(data_path / 'name_and_path.txt', 'r', encoding='utf-8') as file:
            name_n_path = file.read().splitlines()

        return color_anc, color_pos, color_neg, depth_anc, depth_pos, depth_neg, pose_anc, pose_pos, name_n_path

    def load_color(self, path):
        img = np.array(Image.open(path))
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img).astype(np.float32)
        return torch.from_numpy(img)

    def load_depth(self, path):
        img = np.array(Image.open(path))
        img = img[np.newaxis, :, :]
        img = np.ascontiguousarray(img).astype(np.float32)
        return torch.from_numpy(img)
