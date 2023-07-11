import torch
import torch.utils.data.dataset as Dataset
# import my libs
import utils


class TestDataset(Dataset.Dataset):
    def __init__(self, image_dataset, band_list=[1, 2, 3], window_size=(256, 256), stride=128):
        self.image_dataset = image_dataset
        self.band_list = band_list
        self.window_size = window_size
        self.stride = stride
        self.h_window_num, self.w_window_num = utils.get_window_num(self.image_dataset.height, self.image_dataset.width,
                                                                    window_size, stride)
        self.size = self.h_window_num * self.w_window_num

        # calculate mean and std
        # dim: (band_num, 1, 1)
        self.mean, self.std = utils.get_test_mean_std(self.image_dataset, self.band_list)

    def __len__(self):
        # return window num of an image
        return self.size

    def __getitem__(self, idx):
        # read data. image_window: C H W.
        image_window, x_start, y_start = utils.read_idx_pos_data(idx, self.image_dataset, self.w_window_num,
                                                                 self.band_list, self.stride)
        image_window = (image_window - self.mean) / self.std  # normalization
        image_window = torch.from_numpy(image_window).float()  # numpy to tensor

        return image_window, x_start, y_start
