import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.transform import resize
from util import *

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, code_size=128, step=0):
        self.data_dir = data_dir
        self.step = step

        self.normalization = Normalization(mean=0.5, std=0.5)
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        step = self.step

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))[:, :, :3]

        if img.dtype == np.uint8:
            img = img / 255.0

        data = {'input': img}
        data['input'] = resize(data['input'], (128, 128, 3))
        data['label'] = resize(data['input'], (4 * 2 ** step, 4 * 2 ** step, 3))

        data = self.normalization(data)
        data = self.to_tensor(data)

        return data

class mask_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, code_size=128, step=0):
        self.data_dir = data_dir
        self.code_size = code_size
        self.step = step

        # self.normalization = Normalization(mean=0.5, std=0.5)
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        code_size = self.code_size
        step = self.step

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))[:, :]

        mask = {'input': img}
        mask['input'] = resize(img, (code_size, code_size, 3), anti_aliasing=False)

        mask['mask1'] = resize(mask['input'], (4 * 2 ** step, 4 * 2 ** step, 3), anti_aliasing=False)
        
        mask['input'] = mask['input'].astype(np.bool)
        mask['mask1'] = mask['mask1'].astype(np.bool)
        mask['mask2'] = ~mask['mask1']

        # data = self.normalization(data)
        mask = self.to_tensor(mask)

        return mask

class test_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, code_size=128, step=5):
        self.data_dir = data_dir
        self.step = step

        self.normalization = Normalization(mean=0.5, std=0.5)
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        step = self.step

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))[:, :, :3]

        if img.dtype == np.uint8:
            img = img / 255.0

        data = {'input': img}
        data['input'] = resize(data['input'], (128, 128, 3))
        data['mask'] = ~(data['input'].astype(np.bool))

        data = self.normalization(data)
        data = self.to_tensor(data)

        return data

## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):

        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
    
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data