import torch.nn as nn

from src.model.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.model.generator import MultiReceptiveFieldFusion, Generator
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class HiFiGAN(nn.Module):
    def __init__(self, kernel_sizes, rates, initial_channels, dilation_sizes):
        super(HiFiGAN, self).__init__()
        multi_receptive_field_fusion = MultiReceptiveFieldFusion(kernel_sizes, rates, initial_channels, dilation_sizes)
        self.gen = Generator(multi_receptive_field_fusion, len(kernel_sizes), initial_channels)
        self.multi_period_discriminator = MultiPeriodDiscriminator()
        self.multi_scale_discriminator = MultiScaleDiscriminator()
        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())

    def generator(self, mel, audio):
        generated_wavs= self.gen(mel)
        generated_mels = self.mel_spec(generated_wavs)
