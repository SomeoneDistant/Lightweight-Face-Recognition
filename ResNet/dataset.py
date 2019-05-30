import numpy as np
import cv2

import torch
import torch.utils.data as Data
# import augmentation as aug


class Testset(Data.Dataset):

    def __init__(self, file_dir):
        super(Testset, self).__init__()

        self.img_list = []
        self.label_list = []
        file = open(file_dir+'/imglist_iccv.txt', 'r')
        for line in file.read().splitlines():
            data = line.split('\t')
            self.img_list.append(file_dir+data[0][1:])
            self.label_list.append(float(data[1]))

        self.size = len(self.label_list)
        self.num_classes = len(set(self.label_list))

        # self.augmentation = aug.Sequential([
        #     aug.HorizontalFlip(),
        #     aug.ColorWarp(),
        #     aug.GaussionIllumination(),
        #     aug.ContrastAdjust()
        #     aug.GammaAdjust(),
        #     aug.BrightnessAdjust(),
        #     aug.SaturationAdjust(),
        #     aug.HueAdjust(),
        #     aug.RandomScale(),
        #     aug.CenterCrop(),
        #     aug.RandomNoise(),
        #     aug.RandomRotate()
        #     ])

    def __getitem__(self, index):

        img = cv2.imread(self.img_list[index])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))
        img = img / 255
        label = self.label_list[index]
        label = torch.tensor(label).long()

        return img, label

    def __len__(self):
        return self.size
