from torch import nn
from src.models.res_net_components import BottleneckBlock

width_mult = 1
net = nn.Sequential(
    nn.Conv2d(3, 16*width_mult, kernel_size=(3, 3), stride=(2, 2), padding=1),  # output: (64, 62, 62)
    nn.BatchNorm2d(16*width_mult),
    nn.ReLU(),

    # # L2
    BottleneckBlock(16*width_mult, 16*width_mult),
    # BottleneckBlock(16*width_mult, 16*width_mult),
    # BottleneckBlock(16*width_mult, 16*width_mult),

    # # L3
    BottleneckBlock(16*width_mult, 32*width_mult, stride=2),
    # BottleneckBlock(32*width_mult, 32*width_mult),
    # BottleneckBlock(32*width_mult, 32*width_mult),
    # BottleneckBlock(32*width_mult, 32*width_mult),
    # BottleneckBlock(32*width_mult, 32*width_mult),
    # BottleneckBlock(32*width_mult, 32*width_mult),

    # # L4
    BottleneckBlock(32*width_mult, 64*width_mult, stride=2),
    # BottleneckBlock(64*width_mult, 64*width_mult),
    # BottleneckBlock(64*width_mult, 64*width_mult),
    # BottleneckBlock(64*width_mult, 64*width_mult),
    # BottleneckBlock(64*width_mult, 64*width_mult),
    # BottleneckBlock(64*width_mult, 64*width_mult),

    # # L5
    BottleneckBlock(64*width_mult, 128*width_mult, stride=2),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),
    # BottleneckBlock(128*width_mult, 128*width_mult),

    # # L6
    BottleneckBlock(128*width_mult, 256*width_mult, stride=2),
    # BottleneckBlock(256*width_mult, 256*width_mult),
    # BottleneckBlock(256*width_mult, 256*width_mult),
    # BottleneckBlock(256*width_mult, 256*width_mult),
    # BottleneckBlock(256*width_mult, 256*width_mult),
    # BottleneckBlock(256*width_mult, 256*width_mult),
    # BottleneckBlock(256*width_mult, 256*width_mult),


    # nn.AvgPool2d((4, 4)),
    nn.Flatten(),
)