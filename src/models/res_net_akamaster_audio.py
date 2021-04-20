'''
ripped off from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.models.ceclustering import CEClustering
from src.models.mass_clustering import MassClustering

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class AShortcut(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.planes = planes
    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', output_activation=True):
        super(BasicBlock, self).__init__()
        self.use_output_activation = output_activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = AShortcut(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.use_output_activation:
            out = F.relu(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BottleNeckBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', conv_divider = 1, groups = 1, output_activation=True):
        super(BottleNeckBlock, self).__init__()

        self.use_output_activation = output_activation
        norm_layer = nn.BatchNorm2d
        width = int(planes / conv_divider) * groups
        
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width, momentum=0.1)
        self.conv2 = conv3x3(width, width, stride, groups, 1)
        self.bn2 = norm_layer(width, momentum=0.1)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = AShortcut(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

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
            identity = self.shortcut(x)

        out += identity
        if(self.use_output_activation):
            out = F.relu(out)
        out = self.relu(out)
        return out

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
        d = F.avg_pool2d(x, (2, 2))
        p = torch.mul(d, 0)
        p = torch.cat((d, p), dim=1)
        return p
    


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            use_ceclustering=True,
            use_massclustering=False,
            num_classes=10,
            ce_dim_count = 5,
            ce_init_radius = 0.1,
        ):
        super(ResNet, self).__init__()
        self.use_output_cnn = False # instead of using avg/max pool to reduce the output panes to 1x1, use a CNN with matching dimensions
        self.in_planes = 16 # one channel input is expanded into in_panes - this changes the amount of parameters significantly
        self.num_classes= num_classes
        self.ce_dim_count = ce_dim_count
        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # first block has a stride of 1, all others have 2
        resnet_strides = [1] + [2]*(len(num_blocks)-1)
        running_panes = self.in_planes
        self.resnet_layers = nn.Sequential()
        for i, num_block in enumerate(num_blocks):
            # print(f"Adding layer with: block={block}, panes={running_panes}, num_blocks={num_block}")
            self.resnet_layers.add_module(f"ResNetLayer-{i}", self._make_layer(block, running_panes, num_block, resnet_strides[i]))
            running_panes = running_panes * 2

        output_panes = int(running_panes / 2)
        self.layer6 = nn.Conv2d(output_panes, output_panes, kernel_size=(1,2), stride=1, padding=0, bias=False)
        # print(f"classification layer expecting output panes: {output_panes}")
        self.classification = self.make_classification(output_panes, use_ceclustering, use_massclustering)

        self.apply(_weights_init)
        self.save_gradient = False
        self.resnet_gradient = None

        self.save_distance_output = True
        self.distance_output = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            if(i == len(strides) - 1):
                # we want to skip the RELU in the last block, because the classification part of the network does it anyway.
                layers.append(block(self.in_planes, planes, stride, output_activation=True))
            else:
                layers.append(block(self.in_planes, planes, stride, output_activation=True))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _pre_clustering_forward_hook(self, distance_output):
        self.distance_output = distance_output

    def _gradient_hook(self, gradient):
        self.resnet_gradient = gradient
    
    def get_resnet_gradient(self):
        return self.resnet_gradient

    def get_resnet_activations(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.resnet_layers(out)
        return out

    def forward(self, x):
        # print("Running resnet fwd..")
        # print(f"Input shape: {x.shape} --\n{x}")
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.resnet_layers(out)
        
        if self.save_gradient:
            out.register_hook(self._gradient_hook)

        # print(f"Output after resnet layers: {out.shape}--")
        if(self.use_output_cnn):
            out = self.layer6(out)
        else:
            out = F.max_pool2d(out, (out.shape[-2], out.shape[-1]))

        # print(f"Output after avg pool / output cnn: {out.shape}--")

        out = out.view(out.size(0), -1)
        # print(f"Output flattened: {out.shape}--")
        # used for preliminary search implementation:
        with torch.no_grad():
            if self.save_distance_output:
                self.distance_output = out.clone().detach()
        out = self.classification(out)
        # print(f"Output after classification: {out.shape}")
        return out
    
    def make_classification(self, in_features, use_ceclustering, use_massclustering):
        print(f"Creating classification model with the following opts: use cec={use_ceclustering} use mass = {use_massclustering}")
        if use_ceclustering:
            print(f"Creating CEClustering classification model")
            return nn.Sequential(
                # nn.Sigmoid(),
                # nn.Linear(in_features, self.ce_dim_count),

                nn.Sigmoid(),
                CEClustering(
                    # n_dim=self.ce_dim_count,
                    n_dim=in_features,
                    n_clusters=self.num_classes,
                    # init_radius=self.ce_init_radius
                ),
            )
        elif use_massclustering:
            print(f"Creating mass clustering classification model")
            return nn.Sequential(
                # nn.Sigmoid(),
                # nn.Linear(in_features, self.ce_dim_count),

                nn.Sigmoid(),
                MassClustering(
                    # n_dim=self.ce_dim_count,
                    n_dim=in_features,
                    n_clusters=self.num_classes,
                    # init_radius=self.ce_init_radius
                ),
            )
        else:
            return nn.Linear(in_features, self.num_classes)


def resnet20(
    ceclustering = True,
    num_classes = 10,
    init_radius = 10,
    ce_n_dim = 5,
    ) -> ResNet:
    return ResNet(
        BottleNeckBlock, [3, 4, 6, 3],
        use_ceclustering=ceclustering,
        num_classes = num_classes,
        ce_init_radius = init_radius,
        ce_dim_count=ce_n_dim
        )


def resnet32(
    ceclustering=True,
    num_classes = 10,
    init_radius = 0.4,
    ce_n_dim = 5,
    ) -> ResNet:
    return ResNet(
        BottleNeckBlock, [5, 5, 5, 5, 5],
        use_ceclustering=ceclustering,
        num_classes = num_classes,
        ce_init_radius = init_radius,
        ce_dim_count=ce_n_dim
        )


def resnet44(
    ceclustering=True,
    num_classes = 10,
    init_radius = 0.4,
    ce_n_dim = 5,
    ) -> ResNet:
    return ResNet(
        BottleNeckBlock, [7, 7, 7, 7],
        use_ceclustering=ceclustering,
        num_classes = num_classes,
        ce_init_radius = init_radius,
        ce_dim_count=ce_n_dim
        )


def resnet56(
    ceclustering = True,
    num_classes = 10,
    init_radius = 0.2,
    ce_n_dim = 5,
    massclustering = False,
    ) -> ResNet:
    return ResNet(
        BottleNeckBlock, [2, 4, 6, 5, 3],
        use_ceclustering=ceclustering,
        num_classes = num_classes,
        ce_init_radius = init_radius,
        ce_dim_count=ce_n_dim,
        use_massclustering = massclustering,
        )


def resnet110(
    ceclustering = True,
    num_classes = 10,
    init_radius = 0.4,
    ce_n_dim = 5
    ) -> ResNet:
    return ResNet(
        BottleNeckBlock, [18, 18, 18, 18, 18],
        use_ceclustering=ceclustering,
        num_classes = num_classes,
        ce_init_radius = init_radius,
        ce_dim_count=ce_n_dim,
        )


def resnet1202(ceclustering=True, num_classes = 10):
    return ResNet(
        BasicBlock, [200, 200, 200],
        use_ceclustering=ceclustering, num_classes = num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()