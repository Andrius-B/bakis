import torch.nn as nn
import torch.nn.functional as F
import torch
from src.config import Config
import logging

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IdentityDownsample(nn.Module):
    """
    Implements parameter free shortcut connection by identity mapping.
    If dimensions of input x are greater than activations then this
    is rectified by downsampling and then zero padding dimension 1
    as described by option A in paper.

    Parameters:
    - x: tensor
            the input to the block
    """

    def forward(self, x):
        print(f"Downsampling: {x.shape}")
        d = F.avg_pool2d(x, (2, 2))
        print(f"avg_pool2d: {d.shape}")
        p_orig = torch.mul(d, 0)
        p = torch.cat((d, p_orig), dim=1)
        print(f"concatinated: {d.shape} + {p.shape}: {p.shape}")
        return p

# inspired by:
# https://github.com/a-martyn/resnet/blob/master/resnet.py
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
# and ofcourse the great paper: https://arxiv.org/pdf/1512.03385.pdf


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # self.downsample = nn.Sequential(
        #     conv1x1(inplanes, planes*2, stride),
        #     norm_layer(planes*2),
        # )
        self.downsample = IdentityDownsample()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if(x.shape != out.shape):
            identity = self.downsample(x)
        # out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 conv_divider=4, norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # self.downsample = nn.Sequential(
        #     conv1x1(inplanes, planes*2, stride),
        #     norm_layer(planes*2),
        # )
        self.downsample = IdentityDownsample()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        width = int(planes / conv_divider) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, momentum=0.1)
        self.conv2 = conv3x3(width, width, stride, groups, 1)
        self.bn2 = norm_layer(width, momentum=0.1)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if(x.shape != out.shape):
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# NET_SAVE_PATH = "saved_models/mel_2048_conv2d.pth"
# width_mult = 2
# net = nn.Sequential(
#     nn.Conv2d(1, 16*width_mult, kernel_size=(3, 3), stride=(2, 2)),  # output: (64, 62, 62)
#     # nn.BatchNorm2d(64),
#     nn.ReLU(),

#     # L2
#     BottleneckBlock(16*width_mult, 16*width_mult),
#     BottleneckBlock(16*width_mult, 16*width_mult),
#     BottleneckBlock(16*width_mult, 16*width_mult),

#     # L3
#     BottleneckBlock(16*width_mult, 32*width_mult, stride=2),
#     BottleneckBlock(32*width_mult, 32*width_mult),
#     BottleneckBlock(32*width_mult, 32*width_mult),
#     BottleneckBlock(32*width_mult, 32*width_mult),
#     BottleneckBlock(32*width_mult, 32*width_mult),
#     BottleneckBlock(32*width_mult, 32*width_mult),

#     # L4
#     BottleneckBlock(32*width_mult, 64*width_mult, stride=2),
#     BottleneckBlock(64*width_mult, 64*width_mult),
#     BottleneckBlock(64*width_mult, 64*width_mult),
#     BottleneckBlock(64*width_mult, 64*width_mult),
#     BottleneckBlock(64*width_mult, 64*width_mult),
#     BottleneckBlock(64*width_mult, 64*width_mult),

#     # L5
#     BottleneckBlock(64*width_mult, 128*width_mult, stride=2),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),
#     BottleneckBlock(128*width_mult, 128*width_mult),

#     # L6
#     BottleneckBlock(128*width_mult, 256*width_mult, stride=2),
#     BottleneckBlock(256*width_mult, 256*width_mult),
#     BottleneckBlock(256*width_mult, 256*width_mult),
#     BottleneckBlock(256*width_mult, 256*width_mult),
#     BottleneckBlock(256*width_mult, 256*width_mult),
#     BottleneckBlock(256*width_mult, 256*width_mult),
#     BottleneckBlock(256*width_mult, 256*width_mult),


#     nn.AvgPool2d((4, 4)),
#     nn.Flatten(),
# )

