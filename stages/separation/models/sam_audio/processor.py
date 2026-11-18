# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


Anchor = Tuple[str, float, float]


def batch_audio(
    audios: list[str | torch.Tensor], audio_sampling_rate: int = 48_000
) -> Tuple[torch.Tensor, torch.Tensor]:
    wavs = []
    for audio in audios:
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
            if sr != audio_sampling_rate:
                wav = torchaudio.functional.resample(wav, sr, audio_sampling_rate)
        else:
            wav = audio
        wavs.append(wav.mean(0))
    sizes = torch.tensor([wav.size(-1) for wav in wavs])
    return pad_sequence(wavs, batch_first=True).unsqueeze(1), sizes


class Batch:
    def __init__(
        self,
        audios: torch.Tensor,
        sizes: torch.Tensor,
        wav_sizes: torch.Tensor,
        descriptions: list[str],
        hop_length: int,
        audio_sampling_rate: int,
        anchors: Optional[list[list[Anchor]]] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
        masked_video: Optional[torch.Tensor] = None,
    ):
        self.audios = audios
        self.sizes = sizes
        self.wav_sizes = wav_sizes
        self.descriptions = descriptions
        self.audio_pad_mask = audio_pad_mask
        self.masked_video = masked_video
        self.hop_length = hop_length
        self.audio_sampling_rate = audio_sampling_rate
        self.process_anchors(anchors)
        assert self.audios.size(0) == len(self.descriptions)

    def _wav_to_feature_idx(self, wav_idx: int):
        return math.ceil(wav_idx / self.hop_length)

    def to(self, device: torch.device):
        self.audios = self.audios.to(device)
        self.anchor_ids = self.anchor_ids.to(device)
        self.anchor_alignment = self.anchor_alignment.to(device)
        self.sizes = self.sizes.to(device)
        self.wav_sizes = self.wav_sizes.to(device)
        if self.audio_pad_mask is not None:
            self.audio_pad_mask = self.audio_pad_mask.to(device)
        if self.masked_video is not None:
            self.masked_video = [v.to(device) for v in self.masked_video]
        return self

    def process_anchors(self, anchors: Optional[list[list[Anchor]]]):
        batch_size = len(self.audios)
        anchor_dict = {"<null>": 0, "+": 1, "-": 2, "<pad>": 3}
        if anchors is None:
            anchor_ids = torch.full(
                (batch_size, 2), anchor_dict["<null>"], dtype=torch.long
            )
            anchor_ids[:, 1] = anchor_dict["<pad>"]
            anchor_alignment = torch.full(
                (
                    batch_size,
                    self.audio_pad_mask.size(-1),
                ),
                0,
                dtype=torch.long,
            )
            anchor_alignment[~self.audio_pad_mask] = 1  # point to pad token
        else:
            anchor_alignment = torch.full(
                (
                    batch_size,
                    self.audio_pad_mask.size(-1),
                ),
                0,
                dtype=torch.long,
            )
            anchor_alignment[~self.audio_pad_mask] = 1  # point to pad token
            ids = []

            for i, anchor_list in enumerate(anchors):
                current = [anchor_dict["<null>"], anchor_dict["<pad>"]]
                for token, start_time, end_time in anchor_list:
                    start_idx = self._wav_to_feature_idx(
                        start_time * self.audio_sampling_rate
                    )
                    end_idx = self._wav_to_feature_idx(
                        end_time * self.audio_sampling_rate
                    )
                    anchor_alignment[i, start_idx:end_idx] = len(current)
                    current.append(anchor_dict[token])
                ids.append(torch.tensor(current))
            anchor_ids = pad_sequence(
                ids, batch_first=True, padding_value=anchor_dict["<pad>"]
            )
        self.anchor_ids = anchor_ids.to(self.audios.device)
        self.anchor_alignment = anchor_alignment.to(self.audios.device)
        self.anchors = anchors


def mask_from_sizes(sizes: torch.Tensor) -> torch.Tensor:
    return torch.arange(sizes.max()).expand(len(sizes), -1) < sizes.unsqueeze(1)


class SAMAudioProcessor:
    def __init__(self, audio_hop_length, audio_sampling_rate):
        self.audio_hop_length = audio_hop_length
        self.audio_sampling_rate = audio_sampling_rate

    @classmethod
    def from_pretrained(cls, model_dir):
        config = json.loads((Path(model_dir) / "config.json").read_text())["audio_codec"]
        return cls(math.prod(config["encoder_rates"]), config["sample_rate"])

    def wav_to_feature_idx(self, wav_idx):
        return torch.ceil(wav_idx / self.audio_hop_length)

    def __call__(
        self,
        descriptions: list[str],
        audios: list[str | torch.Tensor],
        anchors: Optional[list[list[Anchor]]] = None,
    ):
        """
        Processes input data for the model.

        Args:
            descriptions (list[str]): List of text descriptions corresponding to each audio sample.
            audios (list[str]): List of audio file paths or tensors.
                If a tensor:
                    - should have shape (channels, time) where channels=1 for mono and 2 for stereo.
                    - should be resampled to 48_000 hz
            anchors (Optional[list[list[Anchor]]], optional): List of anchors for each sample,
                where each anchor is a tuple (token, start_time, end_time).

        Returns:
            Batch: A Batch object containing processed audio, sizes, descriptions, anchor ids, anchor alignment, audio pad mask, and optionally masked video.
        """

        assert len(descriptions) == len(audios)
        assert anchors is None or len(descriptions) == len(anchors)

        audios, wav_sizes = batch_audio(audios, self.audio_sampling_rate)

        sizes = self.wav_to_feature_idx(wav_sizes)
        audio_pad_mask = mask_from_sizes(sizes)
        return Batch(
            audios=audios,
            sizes=sizes,
            descriptions=descriptions,
            audio_pad_mask=audio_pad_mask,
            anchors=anchors,
            masked_video=None,
            hop_length=self.audio_hop_length,
            audio_sampling_rate=self.audio_sampling_rate,
            wav_sizes=wav_sizes,
        )

