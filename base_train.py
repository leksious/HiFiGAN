import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
from Losses import FeatureLoss, LossGenerator, MelLoss, DiscriminatorLoss
import numpy as np
import random
from tqdm import tqdm


def train_epoch(config, train_dataloader, generator, optimizer_generator, scheduler_generator, discriminator,
                optimizer_discriminator, scheduler_discriminator, mel_maker, device):
    generator.train()
    discriminator.train()

    loss_a = LossGenerator()
    loss_f = FeatureLoss()
    loss_ms = MelLoss()
    loss_disc = DiscriminatorLoss()

    train_melspec_loss = 0

    for batch in train_dataloader:
        batch = collator(batch, mel_maker, device, for_training=True)

        # учим генератор
        wav_pred = generator(batch.melspec)
        melspec_pred = mel_maker(wav_pred)
        discr_output = discriminator(batch.waveform, wav_pred)

        loss_adv = loss_a(discr_output["gen"])
        loss_fm = loss_f(discr_output["feature_maps_gt"], discr_output["feature_maps_gen"])
        loss_mel = loss_ms(batch.melspec, melspec_pred)
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

        wav_pred = generator(batch.melspec)
        discr_output = discriminator(batch.waveform, wav_pred)
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
            batch = collator(
                batch, mel_maker,
                device, for_training=False
            )
            wav_fake = generator(batch.melspec)
            melspec_pred = mel_maker(wav_fake)

            loss_mel = loss_mel(batch.melspec, melspec_pred)
            val_loss_mel += loss_mel.item()

            wandb.log({
                "Mel Loss Val": loss_mel.item()})
        id = np.random.randint(0, batch.waveform.shape[0])
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


def collator(
        batch,
        mel_maker,
        device: torch.device,
        for_training: bool,
        segment_size: int = 8192
):
    if for_training:
        waveform_segment = []

        for idx in range(batch.waveform.shape[0]):
            waveform_length = batch.waveform_length[idx]
            waveform = batch.waveform[idx][:waveform_length]

            if waveform_length >= segment_size:
                difference = waveform_length - segment_size
                waveform_start = random.randint(0, difference - 1)
                waveform_segment.append(
                    waveform[waveform_start:waveform_start + segment_size]
                )
            else:
                waveform_segment.append(
                    F.pad(waveform, (0, segment_size - waveform_length))
                )

        batch.waveform = torch.vstack(waveform_segment)

    batch.melspec = mel_maker(batch.waveform.to(device))
    batch.melspec_loss = mel_maker(batch.waveform.to(device))

    return batch.to(device)
