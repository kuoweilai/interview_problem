
from pathlib import Path

from PIL import Image

import numpy as np

import torch
import torchvision
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import transforms
from utils.data_aug.gaussian_blur import GaussianBlur
from utils.data_aug.view_generator import ContrastiveLearningViewGenerator

class FFDdataset(Dataset):

    def __init__(self, txt_file, root_dir, split='train'):

        self.root_dir = root_dir
        self.split = split
        self.data_list = []
		
        with open(txt_file, 'r') as file:
            for line in file:
                self.data_list.append(line.rstrip('\n'))

    # https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    def get_simclr_pipeline_transform(self, sz, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = v2.Compose([v2.ToImage(),
                                      v2.ToDtype(torch.float32, scale=True),
                                      transforms.RandomResizedCrop(size=sz),
                                      transforms.RandomHorizontalFlip(),
                                      # transforms.RandomApply([color_jitter], p=0.8),
                                      transforms.RandomGrayscale(p=0.2)])
                                      # GaussianBlur(kernel_size=3)])
        return data_transforms
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        name = self.data_list[idx].split(' ')[0]
        label = self.data_list[idx].split(' ')[1]
        
        img_name = Path(str(self.root_dir) + '/' + self.split + '/' + name)

        image = Image.open(img_name)
        # image_resized = image.resize((32, 32), Image.Resampling.BILINEAR)
        # image = np.asarray(image_resized)
        # image = np.asarray(image)
        # image = image.astype(np.float32) / 256.0
        
        # image = np.moveaxis(image, -1, 0)
        # print(image)

        # image1 = image
        # image2 = image

        transform = self.get_simclr_pipeline_transform(16)
        
        image1 = transform(image)
        image2 = transform(image)
        # print(image1.shape)
        
        return image1, image2, label