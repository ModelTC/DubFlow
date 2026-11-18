"""S3FD network; retains upstream checkpoint keys and caches exact-shape priors."""

import torch
import torch.nn as nn
import torch.nn.init as init
from .utils import Detect, PriorBox


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class S3FDNet(nn.Module):

    def __init__(self):
        super(S3FDNet, self).__init__()
        self.priors = {}

        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, 3, 1, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.ReLU(inplace=True),
        ])

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, padding=1),
        ])

        self.loc = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(1024, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(256, 4, 3, 1, padding=1),
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(1024, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(256, 2, 3, 1, padding=1),
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()

    def forward(self, x):
        net, size = self, x.size()[2:]
        sources = []
        for index in range(16):
            x = net.vgg[index](x)
        sources.append(net.L2Norm3_3(x))
        for index in range(16, 23):
            x = net.vgg[index](x)
        sources.append(net.L2Norm4_3(x))
        for index in range(23, 30):
            x = net.vgg[index](x)
        sources.append(net.L2Norm5_3(x))
        for index in range(30, len(net.vgg)):
            x = net.vgg[index](x)
        sources.append(x)
        for index, layer in enumerate(net.extras):
            x = torch.nn.functional.relu(layer(x), inplace=True)
            if index % 2 == 1:
                sources.append(x)

        locations = [net.loc[0](sources[0]).permute(0, 2, 3, 1).contiguous()]
        confidence = net.conf[0](sources[0])
        background, _ = torch.max(confidence[:, :3], dim=1, keepdim=True)
        confidence = torch.cat((background, confidence[:, 3:]), dim=1)
        scores = [confidence.permute(0, 2, 3, 1).contiguous()]
        for index, feature in enumerate(sources[1:], 1):
            scores.append(net.conf[index](feature).permute(0, 2, 3, 1).contiguous())
            locations.append(net.loc[index](feature).permute(0, 2, 3, 1).contiguous())
        feature_maps = tuple(tuple(value.shape[1:3]) for value in locations)
        key = (tuple(size), feature_maps)
        if key not in self.priors:
            self.priors[key] = PriorBox(size, feature_maps).forward().to(device=x.device, dtype=x.dtype)
        locations = torch.cat([value.view(value.size(0), -1) for value in locations], 1)
        scores = torch.cat([value.view(value.size(0), -1) for value in scores], 1)
        return net.detect.forward(locations.view(locations.size(0), -1, 4),
                                  net.softmax(scores.view(scores.size(0), -1, 2)), self.priors[key])
