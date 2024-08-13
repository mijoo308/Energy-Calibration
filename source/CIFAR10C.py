from torchvision import datasets

import os
import numpy as np
from PIL import Image

import CONFIG

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, level, c_type=None, transform=None, target_transform=None):
        super().__init__(
            root, transform=transform,
            target_transform=target_transform
        )

        self.level_data_num = CONFIG.NUM_CORRUPTED
        self.level_data = []
        self.level_targets = []

        c_data = np.load(os.path.join(root, c_type + '.npy'))
        c_targets = np.load(os.path.join(root, 'labels.npy'))

        start = level*self.level_data_num
        end = (level+1)*self.level_data_num
        l_data  = c_data[start : end]
        l_target = c_targets[start : end]

        self.level_data.append(l_data)
        self.level_targets.append(l_target)

        self.data = np.concatenate(self.level_data)
        self.targets = np.concatenate(self.level_targets)


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        return len(self.data)
