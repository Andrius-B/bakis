from torch import nn


class To2D(nn.Module):
    def forward(self, input):
        # before reshape we have:
        # (batch_size, height, width)
        shape = input.shape
        # we want:
        # (batch_size, num_channels, height, width)
        reshaped = input.view(shape[0], 1, shape[1], shape[2])
        return reshaped


# same as above, but we want to go to 2d -> 1d (linear)
# so the output is: (batch_size, height*width)
class To1D(nn.Module):
    def forward(self, input):
        shape = input.shape
        reshaped = input.view(shape[0], shape[1] * shape[2] * shape[3])
        return reshaped
