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
        self.pool3 = nn.MaxPool2d(5, stride=1, padding=2)
        self.conv3 = conv3x3(num_filters, num_filters)
        self.pool4 = nn.MaxPool2d(5, stride=1, padding=2)
        self.conv4 = conv3x3(num_filters, num_filters)

    def forward(self, x):
        x = self.relu(x)
        p1 = self.pool1(x)
        c1 = self.conv1(p1)
        p2 = self.pool2(c1)
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.conv3(p3)
        p4 = self.pool4(c3)
        c4 = self.conv4(p4)
        return x + c1 + c2 + c3 + c4


class RefineNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.drd4 = conv3x3(2048, 512)
        self.rcu41 = RCU(512, 512)
        self.rcu42 = RCU(512, 512)
        self.crp4 = ChainedPooling(512)
        self.rcu43 = RCU(512, 512)

        self.drd3 = conv3x3(1024, 256)
        self.rcu31big = RCU(256, 256)
        self.rcu32big = RCU(256, 256)
        self.rcu31sml = RCU(512, 512)
        self.rcu32sml = RCU(512, 512)
        self.mrf3big = conv3x3(256, 256)
        self.mrf3sml = conv3x3(512, 256)
        self.crp3 = ChainedPooling(256)
        self.rcu33 = RCU(256, 256)

        self.drd2 = conv3x3(512, 256)
        self.rcu21big = RCU(256, 256)
        self.rcu22big = RCU(256, 256)
        self.rcu21sml = RCU(256, 256)
        self.rcu22sml = RCU(256, 256)
        self.mrf2big = conv3x3(256, 256)
        self.mrf2sml = conv3x3(256, 256)
        self.crp2 = ChainedPooling(256)
        self.rcu23 = RCU(256, 256)

        self.drd1 = conv3x3(256, 256)
        self.rcu11big = RCU(256, 256)
        self.rcu12big = RCU(256, 256)
        self.rcu11sml = RCU(256, 256)
        self.rcu12sml = RCU(256, 256)
        self.mrf1big = conv3x3(256, 256)
        self.mrf1sml = conv3x3(256, 256)
        self.crp1 = ChainedPooling(256)
        self.rcu13 = RCU(256, 256)
        self.rcu14 = RCU(256, 256)
        self.rcu15 = RCU(256, 256)

        self.final = nn.Conv2d(256, num_classes, 1)

    # noinspection PyCallingNonCallable
    def forward(self, inputs):
        x = inputs
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstpool(x)
        x1big = self.encoder1(x)
        x2big = self.encoder2(x1big)
        x3big = self.encoder3(x2big)
        x4big = self.encoder4(x3big)

        # RCU-1
        a4big = self.drd4(x4big)
        a4big = self.rcu41(a4big)
        a4big = self.rcu42(a4big)
        # MRF
        m4 = a4big
        # CRP
        c4 = self.crp4(m4)
        # RCU
        r4 = self.rcu43(c4)

        # RCU-1
        a3big = self.drd3(x3big)
        a3big = self.rcu31big(a3big)
        a3big = self.rcu32big(a3big)
        # RCU-2
        a3sml = self.rcu31sml(r4)
        a3sml = self.rcu32sml(a3sml)
        # MRF
        m3big = self.mrf3big(a3big)
        m3sml = self.mrf3sml(a3sml)
        u3sml = F.upsample(m3sml, scale_factor=2, mode='bilinear')
        m3 = m3big + u3sml
        # CRP
        c3 = self.crp3(m3)
        # RCU
        r3 = self.rcu33(c3)

        # RCU-1
        a2big = self.drd2(x2big)
        a2big = self.rcu21big(a2big)
        a2big = self.rcu22big(a2big)
        # RCU-2
        a2sml = self.rcu21sml(r3)
        a2sml = self.rcu22sml(a2sml)
        # MRF
        m2big = self.mrf2big(a2big)
        m2sml = self.mrf2sml(a2sml)
        u2sml = F.upsample(m2sml, scale_factor=2, mode='bilinear')
        m2 = m2big + u2sml
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
    model = RefineNet(2)

    images = Variable(torch.randn(4, 3, 512, 512), volatile=True)

    model.forward(images)
