import numpy as np
import cv2

import torch
import torch.utils.data as data


class Micro(data.Dataset):

    def __init__(self, file_dir='./micro'):
        super(Micro, self).__init__()

        self.img_list = []
        self.label_list = []
        file = open(file_dir+'/imglist_iccv.txt', 'r')
        for line in file.read().splitlines():
            data = line.split('\t')
            self.img_list.append(file_dir+data[0][1:])
            self.label_list.append(float(data[1]))

        self.size = len(self.label_list)
        self.num_classes = len(set(self.label_list))

    def __getitem__(self, index):

        img = cv2.imread(self.img_list[index])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))

        label = self.label_list[index]
        label = torch.tensor(label).long()

        return img, label

    def __len__(self):
        return self.size
