"""
Created by edwardli on 6/30/21
"""
import torch.nn as nn


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConvolutionalAutoEncoder, self).__init__()
        # define our convolutional auto-encoder here without using convolution layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

        self.latent_enc = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x):
        # implement the encoding pass
        x = self.encoder(x)
        return self.latent_enc(x)

    def decode(self, x):
        # implement the decoding pass
        return self.decoder(x)

    def forward(self, x):
        # forward through the graph and return output tensor
        latent = self.encode(x)
        x = self.decode(latent)
        return x, latent
