import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm

from src.model.utils import calculate_padding
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
                    stride=1,
                    dilation=dilation[i],
                    padding=calculate_padding(kernel_size, dilation[i])
                )
            ) for i in range(conv_layers)]
        )

        self.leaky_relu2 = nn.ModuleList([nn.LeakyReLU(LEAKY_RELU_SLOPE) for _ in range(conv_layers)])
        self.convs2 = nn.ModuleList(
            [weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=calculate_padding(kernel_size, 1)
                )
            ) for i in range(conv_layers)]
        )

    def forward(self, x):
        for lr1, conv1, lr2, conv2 in zip(self.leaky_relu1, self.convs1, self.leaky_relu2, self.convs2):
            xt = conv1(lr1(x))
            xt = conv2(lr2(xt))
            x += xt
        return x

    def remove_weight_norm(self):
        for layer in self.convs1 + self.convs2:
            remove_weight_norm(layer)
        # for layer in self.convs2:
        #     remove_weight_norm(layer)


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, kernel_sizes, rates, initial_channels, dilation_sizes):
        super(MultiReceptiveFieldFusion, self).__init__()
        self.num_upsamples = len(rates)
        self.num_kernels = len(kernel_sizes)
        self.ups = nn.ModuleList()
        for i, (rate, kernel_size) in enumerate(zip(rates, kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(initial_channels // (2 ** i), initial_channels // (2 ** (i + 1)),
                                   kernel_size, rate, padding=(kernel_size - rate) // 2)))
        self.leaky_relus = nn.ModuleList([nn.LeakyReLU(LEAKY_RELU_SLOPE) for _ in range(self.num_upsamples)])

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = initial_channels // (2 ** (i + 1))
            for j, (kernel_size, d) in enumerate(zip(kernel_sizes, dilation_sizes)):
                self.resblocks.append(ResBlock(ch, kernel_size=kernel_size, dilation=d))

    def forward(self, x):
        for i in range(self.num_upsamples):
            x = self.leaky_relus[i](x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        return x


class Generator(nn.Module):
    def __init__(self, mrff: MultiReceptiveFieldFusion, num_kernels: int, initial_channels: int):
        super(Generator, self).__init__()
        self.conv_pre = weight_norm(nn.Conv1d(80, initial_channels, 7, 1, padding=3))
        self.multi_receptive_field_fusion = mrff
        self.leaky_relu_post = nn.LeakyReLU()
        channels = initial_channels // (2 ** num_kernels)
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.multi_receptive_field_fusion(x)
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
