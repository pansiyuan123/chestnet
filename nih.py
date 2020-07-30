import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


object_categories = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def read_object_labels_csv(path_file):
    images = []
    num_categories = len(object_categories)
    print('[dataset] read', path_file)
    with open(path_file, "r") as f:
        for line in f:
            row = line.rstrip().split()
            name = row[0]
            labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
            labels = torch.from_numpy(labels)
            item = (name, labels)
            images.append(item)
    return images


def read_loc_labels_csv(path_file):
    images = []
    num_categories = len(object_categories)
    print('[dataset] read', path_file)
    with open(path_file, "r") as f:
        for line in f:
            row = line.rstrip().split()
            name = row[0]
            labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
            labels = torch.from_numpy(labels)
            ill = object_categories.index(row[num_categories + 1])
            loc = (np.asarray(row[num_categories + 2:])).astype(np.float32)
            loc = torch.from_numpy(loc)
            item = (name, labels, ill, loc)
            images.append(item)
    return images



class ChestXrayDataSet(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, isLoc=False):
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.isLoc = isLoc

        # define path of labels file
        self.path_txt = os.path.join(root, 'data')
        # define filename of csv file
        if set == 'train':
            self.path_file = os.path.join(self.path_txt, '5noise_official_1112.txt')
        else:
            self.path_file = os.path.join(self.path_txt, set + '_official_1112.txt')
        if not os.path.exists(self.path_file):
            raise Exception("The " + set + " labels file is not find!")

        self.classes = object_categories
        if self.isLoc:
            self.images = read_loc_labels_csv(self.path_file)
        else:
            self.images = read_object_labels_csv(self.path_file)

    def __getitem__(self, index):
        if self.isLoc:
            name, target, ill, loc = self.images[index]
        else:
            name, target = self.images[index]
        img_ori = Image.open(os.path.join(self.path_images, name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img_ori)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.isLoc:
            return os.path.join(self.path_images, name), img, target, ill, loc
        else:
            return img, target

    def __len__(self):
        return len(self.images)