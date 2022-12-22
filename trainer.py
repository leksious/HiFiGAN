from HiFiGAN.Discriminator import Discriminator
from HiFiGAN.base_train import train, val_epoch
from HiFiGAN.datasets import Batch, LJSpeechCollator, LJSpeechDataset

import torch
from torch.utils.data import random_split, DataLoader
from HiFiGAN.Generator import Generator
from HiFiGAN.utils import MelSpectrogram
import wandb


def trainer(gen_config, mpd_conf, msd_conf, mel_config, train_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.init(project='HiFi GAN')

    dataset = LJSpeechDataset(train_config.path_to_data)
    train_size = int(0.975 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=LJSpeechCollator(),
        batch_size=train_config.batch_size,
        num_workers=2
    )

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=LJSpeechCollator(),
        batch_size=train_config.batch_size,
        num_workers=2
    )

    generator = Generator(gen_config).to(device)

    discriminator = Discriminator(mpd_conf, msd_conf).to(device)

    mel_maker = MelSpectrogram(mel_config).to(device)

    opt_1 = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, generator.parameters()),
        lr=train_config.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_1 = torch.optim.lr_scheduler.StepLR(opt_1, step_size=1, gamma=0.999)
    opt_2 = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, discriminator.parameters()),
        lr=train_config.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    scheduler_2 = torch.optim.lr_scheduler.StepLR(opt_2, step_size=1, gamma=0.999)

    if train_config.chckp is not None:
        chckp = torch.load('saves/best.pt')
        generator.load_state_dict(chckp['generator'])
        discriminator.load_state_dict(chckp['discriminator'])
        opt_1.load_state_dict(chckp['optimizer_generator'])
        opt_2.load_state_dict(chckp['optimizer_discriminator'])

    train(train_config, train_dataloader, val_dataloader, generator, opt_1, scheduler_1, discriminator,
          opt_2, scheduler_2, mel_maker, device)
