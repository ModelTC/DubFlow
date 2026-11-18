import torch
import torch.nn as nn


class AudioEncoder(nn.Module):

    def __init__(self, features):
        super(AudioEncoder, self).__init__()
        self.features = features
        self.deconv = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1)
        self.conv2 = nn.Conv2d(256, 128, 1, stride=1)

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 9:
                output4 = x
            elif i == 14:
                output8 = x
        output8 = self.deconv(output8)
        cat48 = torch.cat((output4, output8), 1)
        output4 = self.conv1(cat48)
        output4 = self.conv2(output4)

        return output4


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def create_audio_encoder():
    return AudioEncoder(make_layers())
