"""SAM-Audio model adapter."""

import gc
import json
import os
from pathlib import Path

import numpy as np
from runtime.models import get
from stages.postprocess.utils import SAMPLE_RATE


def load_sam(model_dir, text_encoder_dir, device):
    model_dir, text_encoder_dir = Path(model_dir).resolve(), Path(text_encoder_dir).resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MODELSCOPE_OFFLINE="1")
    import torch
    from stages.separation.models.sam_audio import SAMAudio, SAMAudioProcessor
    from stages.separation.models.sam_audio.config import SAMAudioConfig

    def load():
        config = json.loads((model_dir / "config.json").read_text())
        config["text_encoder"] = {**config.get("text_encoder", {}), "name": str(text_encoder_dir)}
        config.update(span_predictor=None, visual_ranker=None, text_ranker=None)
        model = SAMAudio(SAMAudioConfig(**config))
        model.load_state_dict(torch.load(model_dir / "checkpoint.pt", map_location="cpu", weights_only=True), strict=True)
        return model.eval().to(device), SAMAudioProcessor.from_pretrained(str(model_dir))

    return get(("sam_audio", str(model_dir), str(text_encoder_dir), device), load)


class SAMSession:
    def __init__(self, model_dir, text_encoder_dir, device):
        import torch

        self.torch, self.device = torch, device
        self.model, self.processor = load_sam(model_dir, text_encoder_dir, device)
        if self.processor.audio_sampling_rate != SAMPLE_RATE:
            raise ValueError("SAM checkpoint must use the 48 kHz source timeline")

    def separate(self, audio, description, seed, anchors=None):
        if anchors == []:
            return np.zeros_like(audio), audio.copy()
        torch = self.torch
        target, residual = np.empty_like(audio), np.empty_like(audio)
        hop = self.processor.audio_hop_length
        device = torch.device(self.device)
        devices = [device.index or 0] if device.type == "cuda" else []
        # SAM averages input channels. Preserve stereo, reusing exact dual-mono input.
        dual_mono = np.array_equal(audio[:, 0], audio[:, 1])
        for channel in range(1 if dual_mono else 2):
            values = torch.from_numpy(audio[:, channel].copy()).unsqueeze(0)
            if values.shape[-1] < hop:
                values = torch.nn.functional.pad(values, (0, hop - values.shape[-1]))
            kwargs = {} if anchors is None else {"anchors": [anchors]}
            batch = self.processor(audios=[values], descriptions=[description], **kwargs).to(self.device)
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
                separated = self.model.separate(batch)
            for track, prediction in ((target, separated.target[0]), (residual, separated.residual[0])):
                values = prediction.detach().float().cpu().numpy().reshape(-1)
                if len(values) < len(audio) or not np.isfinite(values).all():
                    raise ValueError("SAM returned incomplete or nonfinite target/residual")
                # Codec output is hop-padded; remove only its tail padding, never retime speech.
                track[:, channel] = values[:len(audio)]
        if dual_mono:
            target[:, 1], residual[:, 1] = target[:, 0], residual[:, 0]
        return target, residual

    def close(self):
        del self.model, self.processor
        gc.collect()
        if self.device.startswith("cuda"):
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()
