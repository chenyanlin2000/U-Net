import torch
import torch.utils.data.dataset as Dataset
import rasterio as rio
# import my libs
import utils


class TestDataset(Dataset.Dataset):
    def __init__(self, image_filename, label_filename, band_list=[1, 2, 3], window_size=(256, 256), stride=128,
                 with_label=True):
        self.image_dataset = rio.open(image_filename)
        if with_label:
            self.label_dataset = rio.open(label_filename)
        else:
            self.label_dataset = None
        self.band_list = band_list
        self.window_size = window_size
        self.stride = stride
        self.with_label = with_label
        self.h_window_num, self.w_window_num = utils.get_window_num(self.image_dataset.height, self.image_dataset.width,
                                                                    window_size, stride)
        self.size = self.h_window_num * self.w_window_num

        # calculate mean and std
        self.mean, self.std = utils.get_test_mean_std(self.image_dataset, band_list)

    def __len__(self):
        # return window num of an image
        return self.size

    def __getitem__(self, idx):
        # read data. image_window: C H W. label_window: H W
        image_window, label_window = utils.read_idx_pos_data(idx, self.image_dataset, self.label_dataset,
                                                             self.w_window_num, self.h_window_num, self.band_list,
                                                             self.stride, self.with_label)
        if self.with_label:
            label_window = torch.from_numpy(label_window)

        image_window = torch.from_numpy(image_window).float()

        return image_window, label_window
