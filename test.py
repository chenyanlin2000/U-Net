import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import rasterio as rio
# import my libs
from dataset.dataset_test import TestDataset
from model.U_Net import uNet
from config import cfg


def save_image(filename, data, meta):
    with rio.open(filename, 'w', **meta) as dst:
        dst.write(data)


def test(cfg):
    # define device
    device = cfg.TEST.DEVICE


if __name__ == '__main__':
    cfg.merge_from_file("config//config.yaml")
    test(cfg)
    print("Congratulations, the test is done!\n")
