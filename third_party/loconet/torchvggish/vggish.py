import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        # self.embeddings = nn.Sequential(
        #     nn.Linear(512 * 4 * 6, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 128),
        #     nn.ReLU(True))
        self.deconv = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1)
        self.conv2 = nn.Conv2d(256, 128, 1, stride=1)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x = self.features(x)
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
        # res = self.pool(output4)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        # x = torch.transpose(x, 1, 3)
        # x = torch.transpose(x, 1, 2)
        # x = x.contiguous()
        # x = x.view(x.size(0), -1)

        # return self.embeddings(x)
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


def _vgg():
    return VGG(make_layers())
