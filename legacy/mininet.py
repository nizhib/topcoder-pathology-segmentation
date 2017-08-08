import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class RCU(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.relu1 = nn.SELU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes)
        self.relu2 = nn.SELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = x + residual
        return x


class ChainedPooling(nn.Module):
    def __init__(self, num_filters):
        super().__init__()

        self.relu = nn.SELU(inplace=True)
        self.pool1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.conv1 = conv3x3(num_filters, num_filters)
        self.pool2 = nn.MaxPool2d(5, stride=1, padding=2)
        self.conv2 = conv3x3(num_filters, num_filters)

    def forward(self, x):
        x = self.relu(x)
        p1 = self.pool1(x)
        c1 = self.conv1(p1)
        p2 = self.pool2(c1)
        c2 = self.conv2(p2)
        return x + c1 + c2


class MiniNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2

        self.drd2 = conv3x3(128, 64)
        self.rcu21 = RCU(64, 64)
        self.rcu22 = RCU(64, 64)
        self.crp2 = ChainedPooling(64)
        self.rcu23 = RCU(64, 64)

        self.drd1 = conv3x3(64, 64)
        self.rcu11big = RCU(64, 64)
        self.rcu12big = RCU(64, 64)
        self.rcu11sml = RCU(64, 64)
        self.rcu12sml = RCU(64, 64)
        self.mrf1big = conv3x3(64, 64)
        self.mrf1sml = conv3x3(64, 64)
        self.crp1 = ChainedPooling(64)
        self.rcu13 = RCU(64, 64)
        self.rcu14 = RCU(64, 64)
        self.rcu15 = RCU(64, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    # noinspection PyCallingNonCallable
    def forward(self, inputs):
        x = inputs
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstpool(x)
        x1big = self.encoder1(x)
        x2big = self.encoder2(x1big)

        # RCU-1
        a2big = self.drd2(x2big)
        a2big = self.rcu21(a2big)
        a2big = self.rcu22(a2big)
        # MRF
        m2 = a2big
        # CRP
        c2 = self.crp2(m2)
        # RCU
        r2 = self.rcu23(c2)

        # RCU-1
        a1big = self.drd1(x1big)
        a1big = self.rcu11big(a1big)
        a1big = self.rcu12big(a1big)
        # RCU-2
        a1sml = self.rcu11sml(r2)
        a1sml = self.rcu12sml(a1sml)
        # MRF
        m1big = self.mrf1big(a1big)
        m1sml = self.mrf1sml(a1sml)
        u1sml = F.upsample(m1sml, scale_factor=2, mode='bilinear')
        m1 = m1big + u1sml
        # CRP
        c1 = self.crp1(m1)
        # RCU
        r1 = self.rcu13(c1)
        r1 = self.rcu14(r1)
        r1 = self.rcu15(r1)

        f = self.final(r1)

        return F.upsample(f, scale_factor=4, mode='bilinear')

if __name__ == '__main__':
    model = MiniNet(2)

    images = Variable(torch.randn(4, 3, 512, 512), volatile=True)

    model.forward(images)
