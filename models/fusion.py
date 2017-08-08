import torch.nn as nn


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


class ConvResidualConv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FusionNet(nn.Module):
    def __init__(self, num_classes, num_filters=64):
        super().__init__()
        self.in_dim = 3
        self.out_dim = num_filters
        self.final_out_dim = num_classes
        # act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn = nn.ReLU()

        # encoder

        self.down_1 = ConvResidualConv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = ConvResidualConv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = ConvResidualConv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()

        # bridge

        self.bridge = ConvResidualConv(self.out_dim * 4, self.out_dim * 8, act_fn)

        # decoder

        self.deconv_3 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn)
        self.up_3 = ConvResidualConv(self.out_dim * 4, self.out_dim * 4, act_fn)
        self.deconv_2 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.up_2 = ConvResidualConv(self.out_dim * 2, self.out_dim * 2, act_fn)
        self.deconv_1 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn)
        self.up_1 = ConvResidualConv(self.out_dim, self.out_dim, act_fn)

        # output

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        deconv_3 = self.deconv_3(bridge)
        skip_3 = (deconv_3 + down_3) / 2
        up_3 = self.up_3(skip_3)
        deconv_2 = self.deconv_2(up_3)
        skip_2 = (deconv_2 + down_2) / 2
        up_2 = self.up_2(skip_2)
        deconv_1 = self.deconv_1(up_2)
        skip_1 = (deconv_1 + down_1) / 2
        up_1 = self.up_1(skip_1)

        out = self.out(up_1)

        return out
