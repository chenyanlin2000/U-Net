import os
import numpy as np
import random
import rasterio as rio
import math


# functions for TrainDataset
def get_filename_list(fileFolder_path):
    # function: get every file name in the folder; name only, path not included
    # if you want to get file name and its path, please use: file_path = os.path.join(path_name, file_name)
    # parameter: fileFolder_path: file folder path
    # return: filename list. ex:['git-cheat-sheet.pdf', 'Git.xmind']
    if not os.path.isdir(fileFolder_path):
        return None
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


def get_train_mean_std(random_idx, mean, std, image_dataset, band_list):
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
    # label_data: Numpy ndarray. H W
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


# functions for Test
def get_window_num(height, width, window_size=(256, 256), stride=128):
    # function: get height window num and width window num for a image
    # stride < window_size
    h_window_num = math.ceil(height / stride)
    w_window_num = math.ceil(width / stride)
    if (h_window_num - 2) * stride + window_size[0] >= height:
        h_window_num = h_window_num - 1
    if (w_window_num - 2) * stride + window_size[1] >= width:
        w_window_num = w_window_num - 1

    return h_window_num, w_window_num


def get_start_xy(idx, height, width, w_window_num, window_size=(256, 256), stride=128):
    # function: get start_x and start_y for a idx with specific window size and stride
    # height, width: image size
    # window_size: H W
    # presume idx start from 0
    idx = idx + 1
    if idx % w_window_num == 0:  # the last window of a row
        y_start = min((idx // w_window_num - 1) * stride, height - window_size[0])
        x_start = width - window_size[1]
    else:
        y_start = min(idx // w_window_num * stride, height - window_size[0])
        x_start = (idx % w_window_num - 1) * stride
    return x_start, y_start


def read_idx_pos_data(idx, image_dataset, label_dataset, w_window_num, band_list, window_size=(256, 256), stride=128,
                      with_label=True):
    # function: read random position image data and label data with a specific window size
    # image_dataset: rasterio dataset object
    # label_dataset: rasterio dataset object
    # band_list: list. ex:[1, 2, 3]
    # window_size: tuple. ex:(256,256) height×width
    # return: image data and label data. Numpy ndarray

    x_start, y_start = get_start_xy(idx, image_dataset.height, image_dataset.width, w_window_num, window_size, stride)

    # read data. return: Numpy ndarray
    # image_window: C H W
    # label_window: H W
    image_window = image_dataset.read(window=rio.windows.Window(x_start, y_start, window_size[1], window_size[0]),
                                      indexes=band_list)
    if with_label:
        label_window = label_dataset.read(window=rio.windows.Window(x_start, y_start, window_size[1], window_size[0]),
                                          indexes=1)
    else:
        label_window = None
    return image_window, label_window


def get_test_mean_std(image_dataset, band_list):
    # function: get mean and std of an overview image
    # image_dataset: rasterio dataset object
    # band_list: list.

    data = image_dataset.read(indexes=band_list, out_shape=(image_dataset.height // 10, image_dataset.width // 10))
    image_mean = np.mean(data, axis=(1, 2)).reshape(len(band_list), 1, 1)  # mean of every band
    image_std = np.std(data, axis=(1, 2)).reshape(len(band_list), 1, 1)  # std of every band

    return image_mean, image_std
