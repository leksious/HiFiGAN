import torch.nn as nn
import torch
import torch.nn.functional as F


class SubMPDDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        layers = [nn.utils.weight_norm(nn.Conv2d(
            1, 32,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=((kernel_size - 1) // 2, 0)
        ))]

        channels = [32, 128, 512, 1024]
        for i in range(3):
            layers.append(nn.utils.weight_norm(nn.Conv2d(channels[i], channels[i + 1],
                                                         kernel_size=(kernel_size, 1),
                                                         stride=(stride, 1),
                                                         padding=((kernel_size - 1) // 2, 0))))

        layers.append(nn.utils.weight_norm(nn.Conv2d(1024, 1024, kernel_size=(kernel_size, 1),
                                                     padding=(2, 0))))
        layers.append(nn.utils.weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1),
                                                     padding=(1, 0))))
        self.convolutions = nn.ModuleList(layers)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape

        if seq_len % self.period != 0:
            x = F.pad(x, (0, self.period - seq_len % self.period), 'reflect')
            seq_len += self.period - seq_len % self.period

        x = x.view(batch_size, channels, seq_len // self.period, self.period)

        feature_map = []

        i = 0
        for layer in self.convolutions:
            x = layer(x)
            if i != len(self.convolutions) - 1:
                x = F.leaky_relu(x, 1e-1)
            feature_map.append(x)
            i += 1
        x = torch.flatten(x, 1, -1)

        return x, feature_map


class SubMSDDiscriminator(nn.Module):
    def __init__(self, kernel_size=41):
        super().__init__()

        channels = [128, 256, 512, 1024]

        layers = [nn.utils.weight_norm(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7))]
        stride = [2, 4, 4]

        for i in range(3):
            layers.append(nn.utils.weight_norm(nn.Conv1d(1, channels[i], channels[i + 1], kernel_size, stride[i],
                                                         groups=16, padding=20)))
        layers.append(nn.utils.weight_norm(nn.Conv1d(1, 1024, 1024, 41, groups=16,
                                                     padding=(kernel_size - 1) // 2)))
        layers.append(nn.utils.weight_norm(nn.Conv1d(1, 1024, 1024, kernel_size=5, padding=(kernel_size - 1) // 2)))

        layers.append(nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1)))

        self.convolutions = nn.ModuleList(layers)

        self.mean_pool = nn.AvgPool1d(
            kernel_size=4,
            stride=2,
            padding=2
        )

    def forward(self, x):
        x = self.mean_pool(x)
        feature_map = []

        i = 0
        for layer in self.convolutions:
            x = layer(x)
            if i != len(self.convolutions) - 1:
                x = F.leaky_relu(x, 1e-1)
            feature_map.append(x)
            i += 1
        x = torch.flatten(x, 1, -1)
        return x, feature_map


class Discriminator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mpd = MPD(config)
        self.msd = MSD(config)


class MPD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.periods = config.periods
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.sub_layers = nn.ModuleList([
            SubMPDDiscriminator(period, self.kernel_size, self.stride) for period in self.periods
        ])

    def forward(self, wavs, hat_wavs):
        feature_maps_gt = []
        feature_maps_gen = []
        discriminator_scores_true = []
        discriminator_scores_pred = []
        for sub_layer in self.sub_layers:
            score_real, true_feature = sub_layer(wavs)
            score_gen, gen_feature = sub_layer(hat_wavs)
            feature_maps_gt.append(true_feature)
            feature_maps_gen.append(gen_feature)
            discriminator_scores_true.append(score_real)
            discriminator_scores_pred.append(score_gen)

        return discriminator_scores_true, discriminator_scores_pred, feature_maps_gt, feature_maps_gen


class MSD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.sub_layers = nn.ModuleList([
            SubMSDDiscriminator(self.kernel_size) for i in range(3)
        ])

    def forward(self, wavs, hat_wavs):
        feature_maps_gt = []
        feature_maps_gen = []
        discriminator_scores_true = []
        discriminator_scores_pred = []
        for sub_layer in self.sub_layers:
            score_real, true_feature = sub_layer(wavs)
            score_gen, gen_feature = sub_layer(hat_wavs)
            feature_maps_gt.append(true_feature)
            feature_maps_gen.append(gen_feature)
            discriminator_scores_true.append(score_real)
            discriminator_scores_pred.append(score_gen)

        return discriminator_scores_true, discriminator_scores_pred, feature_maps_gt, feature_maps_gen





