import torch
import torch.utils.data.dataset as Dataset
import os
import numpy as np
import rasterio as rio
import random


def get_filename_list(fileFolder_path):
    # function: get every file name in the folder; name only, path not included
    # if you want to get file name and its path, please use: file_path = os.path.join(path_name, file_name)
    # parameter: fileFolder_path: file folder path
    # return: filename list. ex:['git-cheat-sheet.pdf', 'Git.xmind']
    filename_list = []
    for file_name in os.listdir(fileFolder_path):
        filename_list.append(file_name)
    return filename_list


def read_random_pos_data(image_dataset, label_dataset, band_list, window_size=(256, 256)):
    # function: read random position image data and label data with a specific window size
    # image_dataset: rasterio dataset object
    # label_dataset: rasterio dataset object
    # band_list: list. ex:[1, 2, 3]
    # window_size: tuple. ex:(256,256) height×width
    # return: image data and label data. Numpy ndarray

    x_start = random.randint(0, image_dataset.width - window_size[1] - 1)
    y_start = random.randint(0, image_dataset.height - window_size[0] - 1)

    # read data. return: Numpy ndarray
    # image_window: C H W
    # label_window: H W
    image_window = image_dataset.read(window=rio.windows.Window(x_start, y_start, window_size[1], window_size[0]),
                                      indexes=band_list)
    label_window = label_dataset.read(window=rio.windows.Window(x_start, y_start, window_size[1], window_size[0]),
                                      indexes=1)

    return image_window, label_window


def get_mean_std(random_idx, mean, std, image_dataset, band_list):
    # function: get mean and std of an overview image
    # random_idx: index. int
    # mean, std: dictionary
    # image_dataset: rasterio dataset object
    # band_list: list.

    if random_idx in mean.keys():
        image_mean = mean[random_idx]
        image_std = std[random_idx]
    else:
        # data: C H W
        data = image_dataset.read(indexes=band_list, out_shape=(image_dataset.height // 10, image_dataset.width // 10))
        image_mean = np.mean(data, axis=(1, 2)).reshape(len(band_list), 1, 1)  # mean of every band
        image_std = np.std(data, axis=(1, 2)).reshape(len(band_list), 1, 1)  # std of every band

        mean[random_idx] = image_mean
        std[random_idx] = image_std

    return image_mean, image_std


def data_augmentation(image_data, label_data):
    # function: data augmentation. rotate 90° 180° 270°
    # image_data: Numpy ndarray. C H W
    # lable_data: Numpy ndarray. H W
    image_data = image_data[np.newaxis, :]  # add a new axis.C H W to N C H W
    label_data = label_data[np.newaxis, :]  # add a new axis.H W to N H W
    aug_image = image_data
    aug_label = label_data

    for i in range(1, 4):
        temp_image = np.rot90(image_data, k=i, axes=(2, 3))
        aug_image = np.append(aug_image, temp_image, axis=0)

        temp_label = np.rot90(label_data, k=i, axes=(1, 2))
        aug_label = np.append(aug_label, temp_label, axis=0)

    return aug_image, aug_label


class RS_Dataset(Dataset.Dataset):
    def __init__(self, image_folder, label_folder, band_list=[1, 2, 3], window_size=(256, 256), augmentation=True):
        self.image_list = get_filename_list(image_folder)
        self.label_list = get_filename_list(label_folder)
        self.band_list = band_list
        self.window_size = window_size
        self.augmentation = augmentation
        self.size = len(self.image_list)

        self.mean = {}
        self.std = {}

    def __len__(self):
        # return image nums
        return self.size

    def __getitem__(self, idx):
        # Pick and open a random image
        random_idx = random.randint(0, self.size - 1)
        image_dataset = rio.open(self.image_list[random_idx])
        label_dataset = rio.open(self.label_list[random_idx])

        # calculate image mean and std for normalization
        mean, std = get_mean_std(random_idx, self.mean, self.std, image_dataset, self.band_list)

        # read data. image_window: C H W. label_window: H W
        image_window, label_window = read_random_pos_data(image_dataset, label_dataset, self.band_list,
                                                          self.window_size)
        image_window = (image_window - mean) / std  # data normalization
        if self.augmentation:  # data augmentation
            image_window, label_window = data_augmentation(image_window, label_window)

        # image_window: N C H W
        # label_window: N H W
        return torch.from_numpy(image_window).float(), torch.from_numpy(label_window)
