# Dataset construction based on WHU-OHS
import os
import torch
from torch.utils import data
import numpy as np
from osgeo import gdal
from torchvision.transforms import transforms
# import gdal


def spec_derivative(data):

    data_i = data[0:31, ...]
    data_j = data[1:32, ...]

    data_diff = data_j - data_i

    return data_diff


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image(data_root):
    data_path_train_image = os.path.join(data_root, 'tr', 'image')
    data_path_test_image = os.path.join(data_root, 'ts', 'image')

    train_image_list = []
    train_label_list = []
    test_image_list = []
    test_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_train_image)):
        for name in fnames:
            if is_image_file(name):
                fname = name.split('.')[0] + '.tif'
                image_path = os.path.join(data_path_train_image, fname)
                label_path = image_path.replace('image', 'label')
                assert os.path.exists(label_path)
                assert os.path.exists(image_path)
                train_image_list.append(image_path)
                train_label_list.append(label_path)

    for root, paths, fnames in sorted(os.walk(data_path_test_image)):
        for fname in fnames:
            if is_image_file(fname):
                image_path = os.path.join(data_path_test_image, fname)
                label_path = image_path.replace('image', 'label')
                assert os.path.exists(label_path)
                assert os.path.exists(image_path)
                test_image_list.append(image_path)
                test_label_list.append(label_path)

    assert len(train_image_list) == len(train_label_list)
    assert len(test_image_list) == len(test_label_list)

    return train_image_list, train_label_list, test_image_list, test_label_list


class WHU_OHS_Dataset(data.Dataset):
    def __init__(self, image_file_list, label_file_list, use_3D_input=False, channel_last=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.use_3D_input = use_3D_input
        self.channel_last = channel_last


    def __len__(self):
        return len(self.image_file_list)


    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        name = os.path.basename(image_file)
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

        image = image_dataset.ReadAsArray()
        label = label_dataset.ReadAsArray()

        if(self.channel_last):
            image = image.transpose(1, 2, 0)
            # 将x第二维度挪到第一维上，第三维移到第二维上，原本的第一维移动到第三维上

        # The image patches were normalized and scaled by 10000 to reduce storage cost
        image = torch.tensor(image, dtype=torch.float)
        # 原始 data
        image_oral = image / 10000.0


        # data diff
        data_diff = spec_derivative(image_oral)
        C, H, W = image.shape
        image_diff = torch.zeros((C, H, W))
        image_diff[0:31, ...] = data_diff

        if(self.use_3D_input):
            image = image.unsqueeze(0)
            # 增加第一维

        label = torch.tensor(label, dtype=torch.long) - 1

        return image_oral, image_diff, label, name

