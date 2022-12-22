import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        paddings = [(kernel_size * d - d) // 2 for d in dilation]

        self.convolutions = nn.ModuleList([
            torch.nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, padding=paddings[i],
                                                 dilation=dilation[i]))
            for i in range(len(dilation))
        ])

    def forward(self, x):
        for layer in self.convolutions:
            residual = x
            x = layer(F.leaky_relu(x, 1e-1))
            x += residual
        return x


class MRF(nn.Module):
    def __init__(self, kernel_size, channels, dilation):
        super().__init__()
        self.k_r = len(kernel_size)
        self.blocks = nn.ModuleList([ResBlock(channels, kernel_size[i], dilation[i]) for i in range(self.k_r)])

    def forward(self, x):
        out = 0
        for i, layer in enumerate(self.blocks):
            if i == 0:
                out = layer(x)
            else:
                out += layer(x)
        out /= self.k_r
        return out


class USBlock(nn.Module):
    def __init__(self, channels, init_conv_kernel_size, kernel_size, dilation):
        super().__init__()
        stride = init_conv_kernel_size // 2
        padding = (init_conv_kernel_size - stride) // 2
        out_channels = channels // 2
        self.convolution = nn.utils.weight_norm(nn.ConvTranspose1d(channels, out_channels, init_conv_kernel_size,
                                                                   stride, padding=padding))
        self.mrf = MRF(kernel_size, out_channels, dilation)

    def forward(self, x):
        x = F.leaky_relu(x, 1e-1)
        x = self.convolution(x)
        x = self.mrf(x)
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.p_c = nn.utils.weight_norm(nn.Conv1d(config.mel, config.pre_channels, 7, 1, padding=3))

        self.usb_block = nn.Sequential(*list(
            USBlock(config.pre_channels // 2 ** i, config.kernel_size[i], config.kernel_res, config.dilation)
            for i in range(len(config.kernel_size))
        ))

        self.post_net = nn.utils.weight_norm(nn.Conv1d(config.pre_channels // 2 ** len(config.kernel_size),
                                                       1, 7, 1, padding=3))

    def forward(self, x):
        x = self.p_c(x)

        x = self.usb_block(x)

        x = F.leaky_relu(x)

        x = self.post_net(x)

        waveform_prediction = torch.tanh(x)

        return waveform_prediction

