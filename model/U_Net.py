import torch
import torch.nn as nn


class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(convBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


class upSampling(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(upSampling, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),

            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


class uNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(uNet, self).__init__()
        self.enCode1 = convBlock(in_channels=in_channels, out_channels=64)
        self.enCode2 = convBlock(in_channels=64, out_channels=128)
        self.enCode3 = convBlock(in_channels=128, out_channels=256)
        self.enCode4 = convBlock(in_channels=256, out_channels=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deCode1 = upSampling(in_channels=512, middle_channels=1024, out_channels=512)
        self.deCode2 = upSampling(in_channels=1024, middle_channels=512, out_channels=256)
        self.deCode3 = upSampling(in_channels=512, middle_channels=256, out_channels=128)
        self.deCode4 = upSampling(in_channels=256, middle_channels=128, out_channels=64)

        self.lastLayer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        enc1 = self.enCode1(x)
        enc1_pool = self.Maxpool(enc1)
        enc2 = self.enCode2(enc1_pool)
        enc2_pool = self.Maxpool(enc2)
        enc3 = self.enCode3(enc2_pool)
        enc3_pool = self.Maxpool(enc3)
        enc4 = self.enCode4(enc3_pool)
        enc4_pool = self.Maxpool(enc4)

        dec1 = self.deCode1(enc4_pool)
        dec2 = self.deCode2(torch.cat((dec1, enc4), dim=1))
        dec3 = self.deCode3(torch.cat((dec2, enc3), dim=1))
        dec4 = self.deCode4(torch.cat((dec3, enc2), dim=1))

        out = self.lastLayer(torch.cat((dec4, enc1), dim=1))
        return out
