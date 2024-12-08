import torch.nn as nn

from src.model.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.model.generator import Generator
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class HiFiGAN(nn.Module):
    def __init__(self, kernel_sizes, initial_channels, dilation_sizes):
        super(HiFiGAN, self).__init__()
        self.gen = Generator(kernel_sizes, initial_channels, dilation_sizes)
        self.multi_period_discriminator = MultiPeriodDiscriminator()
        self.multi_scale_discriminator = MultiScaleDiscriminator()
        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())

    def generator(self, target_mel, **batch):
        generated_wavs = self.gen(target_mel)
        generated_mels = self.mel_spec(generated_wavs)
        return dict(
            generated_mels=generated_mels,
            generated_wavs=generated_wavs,
        )

    def discriminator(self, audios, generated_wavs):
        mpd_real = self.multi_period_discriminator(audios, tag="real")
        mpd_gened = self.multi_period_discriminator(generated_wavs, tag="gened")
        msd_real = self.multi_scale_discriminator(audios, tag="real")
        msd_gened = self.multi_scale_discriminator(generated_wavs, tag="gened")
        out = mpd_real | mpd_gened | msd_real | msd_gened
        return out
