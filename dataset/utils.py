import os
import numpy as np
import random
import rasterio as rio
import math
from sklearn.metrics import confusion_matrix


# functions for TrainDataset
def get_file_list(file_dir):
    # function: get every file name in the directory. Name and path
    # if you only want to get file name, please delete: file_path = os.path.join(file_dir, file_name)
    # parameter: fileFolder_path: file folder path
    # return: filename list.
    if not os.path.isdir(file_dir):
        return None
    file_list = []
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        file_list.append(file_path)
    return file_list


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


# functions for TestDataset
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


def read_idx_pos_data(idx, image_dataset, w_window_num, band_list, window_size=(256, 256), stride=128):
    # function: read random position image data with a specific window size
    # image_dataset: rasterio dataset object
    # band_list: list. ex:[1, 2, 3]
    # window_size: tuple. ex:(256,256) height×width
    # return: image data and label data. Numpy ndarray

    x_start, y_start = get_start_xy(idx, image_dataset.height, image_dataset.width, w_window_num, window_size, stride)

    # read data. return: Numpy ndarray
    # image_window: C H W
    image_window = image_dataset.read(window=rio.windows.Window(x_start, y_start, window_size[1], window_size[0]),
                                      indexes=band_list)

    return image_window, x_start, y_start


def get_test_mean_std(image_dataset, band_list):
    # function: get mean and std of an overview image
    # image_dataset: rasterio dataset object
    # band_list: list.

    data = image_dataset.read(indexes=band_list, out_shape=(image_dataset.height // 10, image_dataset.width // 10))
    # mean and std of every band. before reshape: shape = (band_num,1)
    image_mean = np.mean(data, axis=(1, 2)).reshape(len(band_list), 1, 1)
    image_std = np.std(data, axis=(1, 2)).reshape(len(band_list), 1, 1)

    return image_mean, image_std


# functions for test.py
def get_no_extension_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_path(directory, model_name, image_name, extension):
    # function: get a file path to save predict image or confusion matrix
    # directory: directory
    # model_name: name with extension (absolute path)
    # image_name: as above
    # extension: file extension. ".tif" or ".txt"

    # set path
    model_name = get_no_extension_filename(model_name)
    file_name = get_no_extension_filename(image_name) + extension
    path = os.path.join(directory, model_name)
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, file_name)
    return path


def save_image(pre_dir, model_name, image_name, data, meta):
    # function: save an image with meta data
    # predict save directory
    # model_name: name with extension (absolute path)
    # image_name: as above
    # data: pre array

    # get path
    path = get_path(pre_dir, model_name, image_name, ".tif")
    with rio.open(path, 'w', **meta) as dst:
        dst.write(data)


def cal_evaluation_index(lab, pre, LABELS, con_dir, model_name, image_name):
    # function: calculate confusion matrix, p, r, F-score, iou
    # sklearn.metrics.confusion_matrix: input must be 1 dim numpy array.
    # lab、pre: 2 dim numpy array.
    # LABELS: label id. ex: [0,1,2]

    # confusion matrix
    lab = lab.reshape(lab.size)  # 2 dim to 1 dim
    pre = pre.reshape(pre.size)
    cm = confusion_matrix(lab, pre, labels=LABELS)

    # p, r, f-score, iou
    p = np.zeros(len(LABELS))
    r = np.zeros(len(LABELS))
    f_score = np.zeros(len(LABELS))

    for i in range(0, len(LABELS)):
        p[i] = cm[i][i] / np.sum(cm[:, i])
        r[i] = cm[i][i] / np.sum(cm[i, :])
        f_score[i] = 2 * p[i] * r[i] / (p[i] + r[i])

    path = get_path(con_dir, model_name, image_name, ".txt")
    write_results(path, cm, p, r, f_score)
    return cm, p, r, f_score


def write_results(filepath, cm, p, r, f_score):
    # functions: write results in a txt file
    # cm: confusion matrix
    # p r f_score: evaluation index.
    res_text = open(filepath, 'w')
    res_text.write('confusion matrix:\n')
    res_text.write(str(cm) + '\n\n')
    res_text.write('precision:\n')
    res_text.write(str(p) + '\n\n')
    res_text.write('recall:\n')
    res_text.write(str(r) + '\n\n')
    res_text.write('F-score:\n')
    res_text.write(str(f_score) + '\n\n')

    print('confusion matrix:')
    print(cm)
    print('\nprecision:')
    print(p)
    print('\nrecall:')
    print(r)
    print('\nF-score:')
    print(f_score)
