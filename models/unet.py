import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.run = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.run(x)


class ConvBN2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.run = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.run(x)


class UNet(nn.Module):
    def __init__(self, num_classes, num_features=16, num_channels=3):
        super().__init__()

        self.conv1 = ConvBN2(num_channels, num_features)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBN2(num_features, num_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBN2(num_features * 2, num_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBN2(num_features * 4, num_features * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBN2(num_features * 8, num_features * 16)

        self.up4 = nn.ConvTranspose2d(num_features * 16, num_features * 8,
                                      kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = ConvBN2(num_features * 16, num_features * 8)

        self.up3 = nn.ConvTranspose2d(num_features * 8, num_features * 4,
                                      kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = ConvBN2(num_features * 8, num_features * 4)

        self.up2 = nn.ConvTranspose2d(num_features * 4, num_features * 2,
                                      kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = ConvBN2(num_features * 4, num_features * 2)

        self.up1 = nn.ConvTranspose2d(num_features * 2, num_features,
                                      kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = ConvBN2(num_features * 2, num_features)

        self.final = nn.Conv2d(num_features, num_classes, 1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        cc = self.center(p4)

        u4 = self.up4(cc)

        c5 = self.conv5(torch.cat([u4, c4], 1))

        u3 = self.up3(c5)
        c6 = self.conv6(torch.cat([u3, c3], 1))

        u2 = self.up2(c6)
        c7 = self.conv7(torch.cat([u2, c2], 1))

        u1 = self.up1(c7)
        c8 = self.conv8(torch.cat([u1, c1], 1))

        fn = self.final(c8)

        return fn

if __name__ == '__main__':
    model = UNet(2)

    images = Variable(torch.randn(4, 3, 512, 512), volatile=True)

    model.forward(images)
