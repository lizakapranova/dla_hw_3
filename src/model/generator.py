import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm

from src.utils.consts import LEAKY_RELU_SLOPE


class ResBlock(nn.Module):
    def __init__(self, channels: int, conv_layers: int = 3, kernel_size: int = 3,
                 dilation: tuple[int, ...] = (1, 3, 5)):
        super(ResBlock, self).__init__()
        self.channels = channels
        assert len(dilation) == conv_layers

        self.leaky_relu1 = nn.ModuleList([nn.LeakyReLU(LEAKY_RELU_SLOPE) for _ in range(conv_layers)])
        self.convs1 = nn.ModuleList(
            [weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation[i],
                    padding='same'
                )
            ) for i in range(conv_layers)]
        )

    def forward(self, x):
        for lr1, conv1 in zip(self.leaky_relu1,
                              self.convs1):  # lr2, conv2 in zip(self.leaky_relu1, self.convs1, self.leaky_relu2, self.convs2):
            xt = conv1(lr1(x))
            x += xt
        return x

    def remove_weight_norm(self):
        for layer in self.convs1 + self.convs2:
            remove_weight_norm(layer)


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, kernel_sizes, initial_channels, dilation_sizes):
        super(MultiReceptiveFieldFusion, self).__init__()
        self.num_kernels = len(kernel_sizes)

        self.resblocks = nn.ModuleList()
        for kernel_size, d in zip(kernel_sizes, dilation_sizes):
            self.resblocks.append(ResBlock(initial_channels, kernel_size=kernel_size, dilation=d))

    def forward(self, x):
        xs = None
        for i in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[i](x)
            else:
                xs += self.resblocks[i](x)
        return x


class Generator(nn.Module):
    def __init__(self, kernel_sizes, initial_channels: int, dilation_sizes):
        super(Generator, self).__init__()
        self.conv_pre = weight_norm(nn.Conv1d(80, initial_channels, 7, 1, padding=3))
        self.multi_receptive_field_fusion = nn.ModuleList(
            [MultiReceptiveFieldFusion(kernel_sizes, initial_channels // (2 ** (l + 1)), dilation_sizes) for l in
             range(len(kernel_sizes))])
        self.leaky_relu_post = nn.LeakyReLU()
        channels = initial_channels // (2 ** len(kernel_sizes))
        self.leaky_relus = nn.ModuleList([nn.LeakyReLU(LEAKY_RELU_SLOPE) for _ in range(len(kernel_sizes))])
        self.convs = nn.ModuleList([nn.ConvTranspose1d(
            in_channels=(initial_channels // (2 ** l)),
            out_channels=(initial_channels // (2 ** (l + 1))),
            padding=(kernel_sizes[l] - kernel_sizes[l] // 2) // 2,
            kernel_size=kernel_sizes[l],
            stride=(kernel_sizes[l] // 2),
        ) for l in range(len(kernel_sizes))])
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for leaky_relu, conv, mrf in zip(self.leaky_relus, self.convs, self.multi_receptive_field_fusion):
            x = leaky_relu(x)
            x = conv(x)
            x = mrf(x)
        x = self.leaky_relu_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
