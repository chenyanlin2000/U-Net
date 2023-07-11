import torch
import torch.utils.data.dataset as Dataset
import rasterio as rio
import random
# import my libs
import utils


class TrainDataset(Dataset.Dataset):
    def __init__(self, image_folder, label_folder, size=1000, band_list=[1, 2, 3], window_size=(256, 256),
                 augmentation=False):
        self.image_list = utils.get_file_list(image_folder)
        self.label_list = utils.get_file_list(label_folder)
        self.band_list = band_list
        self.window_size = window_size
        self.augmentation = augmentation
        self.size = size

        # Initialize the cache dict
        self.mean = {}
        self.std = {}

    def __len__(self):
        # return image nums
        return self.size

    def __getitem__(self, idx):
        # Pick and open a random image
        random_idx = random.randint(0, len(self.image_list) - 1)
        image_dataset = rio.open(self.image_list[random_idx])
        label_dataset = rio.open(self.label_list[random_idx])

        # calculate image mean and std for normalization
        mean, std = utils.get_train_mean_std(random_idx, self.mean, self.std, image_dataset, self.band_list)

        # read data. image_window: C H W. label_window: H W
        image_window, label_window = utils.read_random_pos_data(image_dataset, label_dataset, self.band_list,
                                                                self.window_size)
        image_window = (image_window - mean) / std  # data normalization
        if self.augmentation:  # data augmentation
            image_window, label_window = utils.data_augmentation(image_window, label_window)

        # with no augmentation:
        # image_window: C H W
        # label_window: H W
        image_window = torch.from_numpy(image_window).float()
        label_window = torch.from_numpy(label_window)
        return image_window, label_window
