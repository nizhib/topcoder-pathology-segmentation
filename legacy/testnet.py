import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class TestNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.final = nn.Conv2d(1792, num_classes, 1)

    # noinspection PyCallingNonCallable
    def forward(self, inputs):
        x = self.firstconv(inputs)
        x = self.firstbn(x)
        e0 = self.firstrelu(x)
        p0 = self.firstpool(e0)
        e1 = self.encoder1(p0)
        e2 = self.encoder2(e1)
        x = F.upsample(e2, scale_factor=2, mode='bilinear')
        x = torch.cat([x, e2], 1)
        x = self.final(x)
        x = F.upsample(x, scale_factor=8, mode='nearest')
        return x

if __name__ == '__main__':
    model = TestNet(2)

    images = Variable(torch.randn(4, 3, 512, 512), volatile=True)

    model.forward(images)
