import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# Implementation of U-Net According to https://arxiv.org/pdf/1505.04597.pdf
# Further work ideas:
#    - Try LeakyReLU instead of ReLU
#    - Increase depth of U structure
#


class Unet(nn.Module):

    def __init__(self,
                 n_classes: int,
                 n_channels: int = 3,
                 base_maps: int = 64):
        super(Unet, self).__init__()

        # Compute contracting blocks
        # Input size of ith block: (I/2**i) - ((2**i - 1)/2**(i-2))
        self._c_out_1 = DoubleConv2d(n_channels, base_maps)
        self._c_out_2 = DoubleConv2d(base_maps, base_maps * 2)
        self._c_out_3 = DoubleConv2d(base_maps * 2, base_maps * 4)
        self._c_out_4 = DoubleConv2d(base_maps * 4, base_maps * 8)
        self._maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Junction between contracting and expanding parts
        self._bottleneck = DoubleConv2d(base_maps * 8, base_maps * 16)

        # Compute expanding blocks
        self._e_input_1 = Upsampling(base_maps * 16, base_maps * 8)
        self._e_input_2 = nn.Sequential(
            DoubleConv2d(base_maps * 16, base_maps * 8),
            Upsampling(base_maps * 8,  base_maps * 4)
        )
        self._e_input_3 = nn.Sequential(
            DoubleConv2d(base_maps * 8, base_maps * 4),
            Upsampling(base_maps * 4,  base_maps * 2)
        )
        self._e_input_4 = nn.Sequential(
            DoubleConv2d(base_maps * 4, base_maps * 2),
            Upsampling(base_maps * 2,  base_maps)
        )

        # Define expanding output layer
        self._output = nn.Sequential(
            DoubleConv2d(base_maps * 2, base_maps),
            nn.Dropout(0.25),
            # 1 x 1 convolution to generate output class number of channels
            nn.Conv2d(in_channels=base_maps,
                      out_channels=n_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, images):
        # Contracting path forward propagation
        c_out_1 = self._c_out_1(images)
        c_out_2 = self._c_out_2(self._maxpool(c_out_1))
        c_out_3 = self._c_out_3(self._maxpool(c_out_2))
        c_out_4 = self._c_out_4(self._maxpool(c_out_3))

        # Bottleneck connects contracting end and expanding start
        bottleneck = self._bottleneck(self._maxpool(c_out_4))
        # Build expanding path input by upsampling
        e_input_1 = self._e_input_1(bottleneck)
        bypass_1 = crop_as(c_out_4, e_input_1)
        e_input_2 = self._e_input_2(torch.cat((bypass_1, e_input_1), axis=1))
        bypass_2 = crop_as(c_out_3, e_input_2)
        e_input_3 = self._e_input_3(torch.cat((bypass_2, e_input_2), axis=1))
        bypass_3 = crop_as(c_out_2, e_input_3)
        e_input_4 = self._e_input_4(torch.cat((bypass_3, e_input_3), axis=1))

        # End of expanding path
        bypass_4 = crop_as(c_out_1, e_input_4)
        return self._output(torch.cat((bypass_4, e_input_4), axis=1))


class DoubleConv2d(nn.Module):

    # Contracting block = Conv2 + Relu + Conv2 + Relu

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv2d, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self._block(x)


class Upsampling(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(Upsampling, self).__init__()
        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2,
                               stride=2)
        )

    def forward(self, x):
        return self._block(x)


def crop_as(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    margin = (ref.shape[2] - tensor.shape[2]) // 2
    return F.pad(tensor, (margin, margin, margin, margin))


def initialize_weights(param):
    class_name = param.__class__.__name__
    if class_name.startswith('Conv'):
        print(f'Initializing weights for layer {class_name}')
        _, in_maps, k, _ = param.weight.shape
        n = k*k*in_maps
        std = np.sqrt(2/n)
        nn.init.normal_(param.weight.data, mean=0.0, std=std)
    else:
        print(f'No need to initialize weights for {class_name}')


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
    model = Unet(n_classes=5).to(device)
    model.apply(initialize_weights)
    input_batch = torch.randn((1, 3, 572, 572)).to(device)
    print(model(input_batch).detach().shape)
