import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from src.model.utils import calculate_padding
from src.utils.consts import LEAKY_RELU_SLOPE


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super(ScaleDiscriminator, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.leaky_relus = nn.ModuleList([nn.LeakyReLU(LEAKY_RELU_SLOPE) for _ in range(len(self.convs))])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv, lrelu in zip(self.convs, self.leaky_relus):
            x = conv(x)
            x = lrelu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x, tag):
        logits, activation_layers = [], []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            logit, alayers = discriminator(x)
            logits.append(logit)
            activation_layers += alayers

        return {
                f"{tag}_msd_activation_layers": activation_layers,
                f"{tag}_msd_logits": logits
            }


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(calculate_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(calculate_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(calculate_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(calculate_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.leaky_relus = nn.ModuleList([nn.LeakyReLU(LEAKY_RELU_SLOPE) for _ in range(len(self.convs))])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv, leaky_relu in zip(self.convs, self.leaky_relus):
            x = conv(x)
            x = leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period=2),
            PeriodDiscriminator(period=3),
            PeriodDiscriminator(period=5),
            PeriodDiscriminator(period=7),
            PeriodDiscriminator(period=11),
        ])

    def forward(self, x, tag):
        logits, activation_layers = [], []
        for discriminator in self.discriminators:
            logit, alayers = discriminator(x)
            logits.append(logit)
            activation_layers += alayers

        return {
            f"{tag}_mpd_activation_layers": activation_layers,
            f"{tag}_mpd_logits": logits
        }
