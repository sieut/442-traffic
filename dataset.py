import cv2
import numpy as np
import os
import random
import torch
from torch.utils.data.dataset import Dataset
from collections import defaultdict


N_CLASS = 2


class TrafficDataset(Dataset):
    def __init__(self, data=None, size=12):
        if data:
            self.data = data
            return

        # Load all images and labels first
        init_data = []
        all_ims = defaultdict(list)
        with open("./data/labels") as file:
            for line in file:
                im_id, label_ = line.split(",")
                imfile = "./data/%s/im_%s" % tuple(im_id.split("_"))

                im = cv2.imread(imfile)
                im = np.asarray(im).astype("f").transpose(2, 0, 1)
                all_ims[im_id.split("_")[0]].append(np.copy(im))

                label_ = int(label_)
                label = np.zeros((N_CLASS, 1))
                label[label_] = 1

                init_data.append((im, label, imfile))

                if len(init_data) == size:
                    break

        # Calculate averages for images with same cam_id
        avg_imgs = {
            cam_id: np.array([np.array(ims)[:,i,:,:].mean(axis=0) for i in range(3)])
            for cam_id, ims in all_ims.items()
        }

        # Subtract the average from loaded images and normalize
        self.data = []
        for data in init_data:
            im, label, imfile = data
            normalized = (im - avg_imgs[imfile.split("/")[2]])
            normalized = normalized / (normalized.max() - normalized.min() / 2) - 1.0
            self.data.append((normalized, label, imfile))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        im, label, _ = self.data[index]
        im = torch.FloatTensor(im)
        label = torch.FloatTensor(label)
        label = torch.LongTensor(torch.argmax(label, dim=0))
        return im, label

    def split(self, train_size=0.8, test_size=0.15):
        """Split Dataset into 3 Datasets, for train, test and validation"""
        random.shuffle(self.data)
        train_cutoff = int(len(self.data) * train_size)
        test_cutoff = train_cutoff + int(len(self.data) * test_size)
        return (
            TrafficDataset(self.data[:train_cutoff]),
            TrafficDataset(self.data[train_cutoff:test_cutoff]),
            TrafficDataset(self.data[test_cutoff:]))