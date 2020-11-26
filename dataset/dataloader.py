#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



class ImageSourceList(Dataset):
    def __init__(self, source_list, labels=None, transform=None, target_transform=None, mode='RGB', batch_number=16):
        imgs = []
        bigest = 0
        for name in source_list:
            image_list = open(name).readlines()
            random.shuffle(image_list)
            if len(image_list) > bigest:
                  bigest = len(image_list)
            img = make_dataset(image_list, labels)
            if len(img) == 0:
                raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
            imgs.append(img)
        for i in range(len(source_list)):
             imgs[i] = imgs[i] * round(0.5 + bigest/len(imgs[i]))
        self.domainnum = len(source_list)
        #self.batchnum = batch_number
        self.bigest = bigest 
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        #from IPython import embed;embed();exit();
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        #i,j = index//self.domainnum, index%self.domainnum
        j = random.randint(0, self.domainnum-1)
        path, target = self.imgs[j][index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, j

    #def __getitem__(self, index):
    #    imgs=[]
    #    targets=[]
    #    for i in range(self.domainnum):
    #        path, target = self.imgs[i][index]
    #        img = self.loader(path)
    #        if self.transform is not None:
    #            img = self.transform(img)
    #        if self.target_transform is not None:
    #            target = self.target_transform(target)
    #        imgs.append(img)
    #        targets.append(target)
    def __len__(self):
        return self.bigest



