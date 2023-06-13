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

    # 1. load files
    image_list = utils.get_filename_list(cfg.TEST_IMAGE_FOLDER)
    label_list = utils.get_filename_list(cfg.TEST_LABEL_FOLDER)

    # load model
    my_net = torch.load(cfg.TEST.MODEL_LOAD_PATH, map_location=lambda storage, loc: storage)
    my_net.eval()
    my_net.to(device)

    # start to test
    for f in range(0, len(image_list)):
        image_dataset = rio.open(image_list[f])
        meta = image_dataset.meta.copy()
        pre_array = torch.zeros((cfg.DATASET.CLASS_NUM, image_dataset.height, image_dataset.width))
        if cfg.DATASET.WITH_LABEL:
            label_dataset = rio.open(label_list[f])
        else:
            label_dataset = None

        data = TestDataset(image_dataset, label_dataset, cfg.DATASET.BAND_LIST, cfg.DATASET.WINDOW_SIZE,
                           cfg.TEST.STRIDE, cfg.DATASET.WITH_LABEL)
        my_dataloader = DataLoader(data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

        for idx, images, labels, x_start, y_start in enumerate(my_dataloader, 0):
            images = images.to(device)  # batch_size C H W
            if cfg.DATASET.WITH_LABEL:
                labels = labels.to(device)  # batch_size H W

            with torch.no_grad():  # 禁用梯度计算
                outputs = my_net(images)  # num c h w
            outputs = outputs.cpu()

            y_end = y_start + cfg.DATASET.WINDOW_SIZE[0]
            x_end = x_start + cfg.DATASET.WINDOW_SIZE[1]
            for i in range(0, cfg.TEST.BATCH_SIZE):
                pre_array[:, y_start[i]:y_end[i], x_start[i]:x_end[i]] += outputs[i]

            del outputs
            torch.cuda.empty_cache()

        _, pre_array = torch.max(pre_array, 0)
        del _
        pre_array = torch.unsqueeze(pre_array.to(torch.uint8), 0)
        meta.update(
            {
                'count': 1,
                'compress': 'lzw',
            }
        )
        utils.save_image(cfg.TEST.PREDICT_SAVE_PATH, pre_array.numpy(), meta)

        # if test data is with label
        # calculate and output confusion matrix
        # if cfg.DATASET.WITH_LABEL:


if __name__ == '__main__':
    cfg.merge_from_file("config//config.yaml")
    test(cfg)
    print("Congratulations, the test is done!\n")
