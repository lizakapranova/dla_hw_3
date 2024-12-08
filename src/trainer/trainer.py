from itertools import chain

import torch

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.model.hifi_gan import HiFiGAN
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(
            self,
            model: HiFiGAN,
            criterion,
            metrics,
            generator_optimizer,
            discriminator_optimizer,
            generator_scheduler,
            discriminator_scheduler,
            config,
            device,
            dataloaders,
            logger,
            writer,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model,
                         criterion,
                         metrics,
                         generator_optimizer,
                         discriminator_optimizer,
                         generator_scheduler,
                         discriminator_scheduler,
                         config, device, dataloaders, logger, writer)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        if self.len_epoch == 1:
            self.log_step = 1
        else:
            self.log_step = 50

        print('self.log_step:', self.log_step)
        print('self.len_epoch:', self.len_epoch)

        metric_keys = ["G_loss", "mel_loss", "fm_loss", "adv_loss", "D_loss"]
        self.train_metrics = MetricTracker(
            "G_grad_norm", "D_grad_norm", *[m for m in metric_keys], writer=self.writer
        )

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.discriminator_optimizer.zero_grad()

        gen_outputs = self.model.generator(**batch)
        batch.update(gen_outputs)

        dis_outputs = self.model.discriminator(audios=batch['target_mel'], generated_wavs=batch['gened_wavs'].detach())
        batch.update(dis_outputs)

        discriminator_loss = self.criterion.discriminator_loss(**batch)
        batch.update(discriminator_loss)
        discriminator_loss["discriminator_loss"].backward()
        self.discriminator_optimizer.step()
        batch["discriminator_grad_norm"] = self._get_grad_norm(chain(self.model.multi_scale_discriminator.parameters(), self.model.multi_period_discriminator.parameters()))

        # G optimizing
        self.generator_optimizer.zero_grad()

        discriminator_outputs = self.model.discriminator(**batch)
        batch.update(discriminator_outputs)

        generator_loss = self.criterion.generator_loss(**batch)
        batch.update(generator_loss)
        generator_loss["generator_loss"].backward()
        self.generator_optimizer.step()
        batch["generator_grad_norm"] = self._get_grad_norm(self.model.gen.parameters())

        for k, v in batch.items():
            if "loss" in k or "grad_norm" in k:
                metrics.update(k, v.item())

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
