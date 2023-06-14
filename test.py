import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import rasterio as rio
# import my libs
from dataset.dataset_test import TestDataset
from model.U_Net import uNet
from config import cfg
from .dataset import utils


def test(cfg):
    # define device
    device = cfg.TEST.DEVICE

    # 1.load files. file path and name
    image_list = utils.get_file_list(cfg.TEST_IMAGE_DIR)
    label_list = utils.get_file_list(cfg.TEST_LABEL_DIR)
    model_list = utils.get_file_list(cfg.MODEL_LOAD_DIR)

    # 2.load model
    for m in range(0, len(model_list)):
        my_net = torch.load(model_list[m], map_location=lambda storage, loc: storage)
        my_net.eval()
        my_net.to(device)

        # start to test
        for f in range(0, len(image_list)):
            image_dataset = rio.open(image_list[f])
            meta = image_dataset.meta.copy()
            pre_array = torch.zeros((cfg.DATASET.CLASS_NUM, image_dataset.height, image_dataset.width))
            if cfg.DATASET.WITH_LABEL:
                label_dataset = rio.open(label_list[f])
                label_data = label_dataset.read(indexes=1)  # numpy array: H W
            else:
                label_data = None

            data = TestDataset(image_dataset, cfg.DATASET.BAND_LIST, cfg.DATASET.WINDOW_SIZE, cfg.TEST.STRIDE)
            my_dataloader = DataLoader(data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

            for idx, images, x_start, y_start in enumerate(my_dataloader, 0):
                images = images.to(device)  # batch_size C H W
                with torch.no_grad():  # forbid auto grad
                    outputs = my_net(images)  # batch_size c h w
                outputs = outputs.cpu()

                y_end = y_start + cfg.DATASET.WINDOW_SIZE[0]
                x_end = x_start + cfg.DATASET.WINDOW_SIZE[1]
                for i in range(0, cfg.TEST.BATCH_SIZE):
                    pre_array[:, y_start[i]:y_end[i], x_start[i]:x_end[i]] += outputs[i]

                del outputs
                torch.cuda.empty_cache()

            _, pre_array = torch.max(pre_array, 0)  # return pre_array now is a tensor with 2 dim: H W
            del _
            pre_array = torch.unsqueeze(pre_array.to(torch.uint8), 0)  # add a dim: H W to C H W
            pre_array = pre_array.numpy()
            meta.update(
                {
                    'count': 1,
                    'compress': 'lzw',
                }
            )
            utils.save_image(cfg.TEST.PREDICT_SAVE_DIR, model_list[m], image_list[f], pre_array, meta)

            # if test data is with label
            # calculate and output confusion matrix
            if cfg.DATASET.WITH_LABEL:
                utils.cal_evaluation_index(label_data, pre_array[0], cfg.DATASET.LABELS)

            del pre_array


if __name__ == '__main__':
    cfg.merge_from_file("config//config.yaml")
    test(cfg)
    print("Congratulations, the test is done!\n")
