import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class PSPDecoder(nn.Module):
    def __init__(self, in_features, out_features, downsize, upsize):
        super().__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(downsize, downsize),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Upsample(upsize, mode='bilinear')
        )

    def forward(self, x):
        return self.features(x)


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        for m in self.encoder3.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1
        for m in self.encoder4.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1

        self.decoder_a = PSPDecoder(512, 1, 64, 64)
        self.decoder_b = PSPDecoder(512, 1, 32, 64)
        self.decoder_c = PSPDecoder(512, 1, 16, 64)
        self.decoder_d = PSPDecoder(512, 1, 8, 64)

        self.final = nn.Sequential(
            nn.Conv2d(516, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, num_classes, 1),
        )

    # noinspection PyCallingNonCallable
    def forward(self, inputs):
        x = self.firstconv(inputs)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        ax = self.decoder_a(x)
        bx = self.decoder_b(x)
        cx = self.decoder_c(x)
        dx = self.decoder_d(x)
        x = torch.cat([x, ax, bx, cx, dx], 1)
        x = self.final(x)
        return F.upsample(x, inputs.size()[2:], mode='bilinear')

if __name__ == '__main__':
    model = PSPNet(2)

    images = Variable(torch.randn(4, 3, 512, 512), volatile=True)

    model.forward(images)
