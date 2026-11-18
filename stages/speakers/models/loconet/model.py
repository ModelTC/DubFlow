import torch
import torch.nn as nn

from .attention import AttentionLayer
from .convolution import ContextConv
from .audio import create_audio_encoder
from .visual import VisualFrontend, VisualConv1D, VisualTCN


class LoCoNetEncoder(nn.Module):

    def __init__(self, num_speakers=3, num_layers=3, adjust_attention=False):
        super(LoCoNetEncoder, self).__init__()
        self.adjust_attention = adjust_attention
        # Visual Temporal Encoder
        self.visualFrontend = VisualFrontend()    # Visual Frontend
        self.visualTCN = VisualTCN()    # Visual Temporal Network TCN
        self.visualConv1D = VisualConv1D()    # Visual Temporal Network Conv1d

        # Complete ASD checkpoints include the audio encoder; no network downloads.
        self.audioEncoder = create_audio_encoder()
        self.audio_pool = nn.AdaptiveAvgPool1d(1)

        # Audio-visual Cross Attention
        self.crossA2V = AttentionLayer(d_model=128, nhead=8)
        self.crossV2A = AttentionLayer(d_model=128, nhead=8)

        # Audio-visual Self Attention

        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(ContextConv(num_speakers))
            layers.append(AttentionLayer(d_model=256, nhead=8))
        self.convAV = layers

    def forward_visual_frontend(self, x):

        B, T, W, H = x.shape
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1, 2)
        return x

    def forward_audio_frontend(self, x):
        t = x.shape[-2]
        numFrames = t // 4
        pad = 8 - (t % 8)
        x = torch.nn.functional.pad(x, (0, 0, 0, pad), "constant")
        x = self.audioEncoder(x)

        b, c, t2, freq = x.shape
        x = x.view(b * c, t2, freq)
        x = self.audio_pool(x)
        x = x.view(b, c, t2)[:, :, :numFrames]
        x = x.permute(0, 2, 1)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src=x1, tar=x2, adjust=self.adjust_attention)
        x2_c = self.crossV2A(src=x2, tar=x1, adjust=self.adjust_attention)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2, b=1, s=1):
        x = torch.cat((x1, x2), 2)    # B*S, T, 2C
        for i, layer in enumerate(self.convAV):
            if i % 2 == 0:
                x, b, s = layer(x, b, s)
            else:
                x = layer(src=x, tar=x)

        x = torch.reshape(x, (-1, 256))
        return x
