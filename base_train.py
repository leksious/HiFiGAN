import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
# from HiFiGAN.Losses import FeatureLoss, LossGenerator, MelLoss, DiscriminatorLoss
from HiFiGAN.Losses import AdversarialLoss, FeatureMatchingLoss, MelSpectrogramLoss, DiscriminatorLoss

import numpy as np
import random
from tqdm import tqdm


def train_epoch(config, train_dataloader, generator, optimizer_generator, scheduler_generator, discriminator,
                optimizer_discriminator, scheduler_discriminator, mel_maker, device):
    generator.train()
    discriminator.train()

    loss_a = AdversarialLoss()
    loss_f = FeatureMatchingLoss()
    loss_ms = MelSpectrogramLoss()
    loss_disc = DiscriminatorLoss()

    train_melspec_loss = 0

    for batch in train_dataloader:

        mel_gt = batch.melspec.to(device)
        wav_gt = batch.waveform.to(device)

        # учим генератор
        wav_pred = generator(mel_gt)
        melspec_pred = mel_maker(wav_pred).squeeze(1)

        if wav_gt.size(-1) > wav_pred.size(-1):
            pad = wav_gt.size(-1) - wav_pred.size(-1)
            wav_pred = F.pad(wav_pred, (0, pad))
        else:
            pad = wav_pred.size(-1) - wav_gt.size(-1)
            wav_gt = F.pad(wav_gt, (0, pad))

        if melspec_pred.size(-1) > mel_gt.size(-1):
            pad = melspec_pred.size(-1) - mel_gt.size(-1)
            melspec_pred = melspec_pred[:, :, 0:mel_gt.size(-1) - pad + 1]
        discr_output = discriminator(wav_gt, wav_pred)

        loss_adv = loss_a(discr_output["gen"], )
        loss_fm = loss_f(discr_output["feature_maps_gt"], discr_output["feature_maps_gen"])
        loss_mel = loss_ms(mel_gt, melspec_pred)
        train_melspec_loss += loss_mel.item()

        loss_gen = loss_adv + 2 * loss_fm + 45 * loss_mel
        loss_gen.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), config.grad_norm_clip)

        wandb.log({
            "Gen Loss on train": loss_adv.item(),
            "FM Loss train": loss_fm.item(),
            "Mel Loss train": loss_mel.item(),
            "Generator Loss train": loss_gen.item(),
            "LR generator": optimizer_generator.param_groups[0]['lr']
        })
        optimizer_generator.step()

        # учим дискриминатор
        optimizer_discriminator.zero_grad()

        wav_pred = generator(mel_gt)
        discr_output = discriminator(wav_gt, wav_pred)
        discr_loss = loss_disc(discr_output['g_t'], discr_output['gen'])
        discr_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_norm_clip)

        wandb.log({
            "Discriminator Loss Train": discr_loss.item(),
            "Discriminator Learning Rate": optimizer_discriminator.param_groups[0]['lr']
        })

        optimizer_discriminator.step()

    scheduler_generator.step()
    scheduler_discriminator.step()

    return train_melspec_loss / len(train_dataloader)


def val_epoch(config, val_dataloader, generator, discriminator, mel_maker, device):
    generator.eval()
    discriminator.eval()

    loss_mel = MelLoss()

    torch.cuda.empty_cache()

    val_loss_mel = 0

    with torch.no_grad():
        for batch in val_dataloader:
            mel_gt = batch.melspec.to(device)
            wav_gt = batch.waveform.to(device)
            wav_fake = generator(mel_gt)
            melspec_pred = mel_maker(wav_fake)

            loss_mel = loss_mel(mel_gt, melspec_pred)
            val_loss_mel += loss_mel.item()

            wandb.log({
                "Mel Loss Val": loss_mel.item()})
        id = np.random.randint(0, mel_gt.shape[0])
        wandb.log({
            "GT Spec": wandb.Image(batch.melspec[id].detach().cpu(), caption=batch.transcript[id].capitalize()),
            "Pred Spec": wandb.Image(melspec_pred[id].detach().cpu(), caption=batch.transcript[id].capitalize())})

    return val_loss_mel / len(val_dataloader)


def train(config, train_dataloader, val_dataloader, generator, optimizer_generator, scheduler_generator, discriminator,
          optimizer_discriminator, scheduler_discriminator, mel_maker, device):
    loss_history = []
    epoch = 0

    for epoch in tqdm(range(config.num_epoch)):

        train_loss = train_epoch(
            config, train_dataloader,
            generator, optimizer_generator, scheduler_generator,
            discriminator, optimizer_discriminator, scheduler_discriminator,
            mel_maker, device)

        val_loss = val_epoch(config, val_dataloader, generator, discriminator, mel_maker, device)

        loss_history.append(val_loss)

        wandb.log({'epoch': epoch,
                   'Loss Train': train_loss,
                   'Loss_val': val_loss,
                   })

        if val_loss <= min(loss_history):
            state = {
                "generator": generator.state_dict(),
                "optimizer_generator": optimizer_generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_discriminator": optimizer_discriminator.state_dict(),
                "config": config
            }
            torch.save(state, config.path_to_save + "/best.pt")
