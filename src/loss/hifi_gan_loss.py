import torch
from torch import nn
from torch.nn.functional import l1_loss


class HiFiGANLoss(nn.Module):
    def __init__(self, fm_loss_weight: float, mel_loss_weight: float, **kwargs):
        super().__init__()
        self.fm_loss_weight = fm_loss_weight
        self.mel_loss_weight = mel_loss_weight

    def mel_loss(self, fake_mels, real_mels, **batch) -> dict:
        return {"mel_loss": l1_loss(fake_mels, real_mels)}

    def fm_loss(self, real_msd_activations, fake_msd_activations, real_mpd_activations, fake_mpd_activations,
                **batch) -> dict:
        fm_loss = 0.0
        for real, fake in zip(real_mpd_activations, fake_mpd_activations):
            fm_loss = fm_loss + l1_loss(fake, real)
        for real, fake in zip(real_msd_activations, fake_msd_activations):
            fm_loss = fm_loss + l1_loss(fake, real)
        return {"fm_loss": fm_loss}

    def adv_loss(self, fake_msd_logits, fake_mpd_logits, **batch) -> dict:
        adv_loss = 0.0
        for logits in fake_msd_logits:
            adv_loss = adv_loss + torch.mean(torch.square(logits - 1))
        for logits in fake_mpd_logits:
            adv_loss = adv_loss + torch.mean(torch.square(logits - 1))
        return {"adv_loss": adv_loss}

    def generator_loss(self, **batch) -> dict:
        out = dict()
        out.update(self.fm_loss(**batch))
        out.update(self.adv_loss(**batch))
        out.update(self.mel_loss(**batch))
        out["generator_loss"] = out["adv_loss"] + self.fm_loss_weight * out["fm_loss"] + self.mel_loss_weight * out[
            "mel_loss"]
        return out

    def discriminator_loss(self, real_msd_logits, fake_msd_logits, real_mpd_logits, fake_mpd_logits, **batch) -> dict:
        discriminator_loss = 0.0
        for real_logits, fake_logits in zip(real_msd_logits, fake_msd_logits):
            discriminator_loss = discriminator_loss + torch.mean(torch.square(real_logits - 1)) + torch.mean(
                torch.square(fake_logits))
        for real_logits, fake_logits in zip(real_mpd_logits, fake_mpd_logits):
            discriminator_loss = discriminator_loss + torch.mean(torch.square(real_logits - 1)) + torch.mean(
                torch.square(fake_logits))
        return {"discriminator_loss": discriminator_loss}
