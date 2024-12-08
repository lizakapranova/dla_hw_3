import warnings
from itertools import chain

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="hifi_gan")
def main(config):
    """
    Main script for training. Instantiates the model, generator_optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build generator_optimizer, learning rate scheduler
    generator_trainable_params = filter(lambda p: p.requires_grad, model.gen.parameters())
    discriminator_trainable_params = filter(lambda p: p.requires_grad, chain(model.msd.parameters(), model.mpd.parameters()))

    generator_optimizer = config.init_obj(config["generator_optimizer"], torch.optim, generator_trainable_params)
    discriminator_optimizer = config.init_obj(config["discriminator_optimizer"], torch.optim, discriminator_trainable_params)

    generator_scheduler = config.init_obj(config["generator_scheduler"], torch.optim.lr_scheduler, generator_optimizer)
    discriminator_scheduler = config.init_obj(config["discriminator_scheduler"], torch.optim.lr_scheduler, discriminator_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(model=model, criterion=loss_function, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                      generator_scheduler=generator_scheduler, discriminator_scheduler=discriminator_scheduler, config=config, device=device,
                      dataloaders=dataloaders, logger=logger, writer=writer, skip_oom=config.trainer.get("skip_oom", True))

    trainer.train()


if __name__ == "__main__":
    main()
