"""Translation prompts and local/API speech model adapters."""

import io
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from runtime.clients import SpeechSynthesizer
from runtime.execution import call_function
from runtime.languages import LANGUAGES
from runtime.models import get
from runtime.utils import file_signature as signature

QWEN_TTS_FILES = ["config.json", "model.safetensors", "speech_tokenizer/config.json",
                  "speech_tokenizer/model.safetensors"]


def translation_prompt(record, conversation, source_language, target_language, shorten=False):
    seconds = record.get("available_seconds", record["window_end"] - max(record["window_start"], record["start"]))
    position = next((index for index, item in enumerate(conversation)
                     if item["id"] == record.get("id")), len(conversation))
    neighbors = conversation[max(0, position - 2):position + 3]
    previous = next((item for item in reversed(conversation[:position])
                     if record.get("speaker_id") is not None and item["speaker_id"] == record["speaker_id"]), None)
    if previous is not None and previous not in neighbors:
        neighbors = [previous, *neighbors]
    context = "\n".join(f"{item['id']} ({item['speaker_id']}): {item['text']}"
                        + (f" [Existing translation: {item['target_text']}]" if item.get("target_text") else "")
                        for item in neighbors if item["id"] != record.get("id"))
    continuity = (
        f"[Background Information]\n{context}\n\n"
        "Keep each speaker's tone, level of formality, forms of address and relationship consistent across turns. "
        "Use that speaker's previous dialogue and the source context; do not invent a relationship or copy "
        "another speaker's register. Shortening must not change these choices just to meet the time budget. "
        "Do not borrow another speaker's words or any other line from the background, even from the same speaker.\n"
    )
    if shorten:
        prompt = continuity + (
            f"Translate ONLY the {LANGUAGES[source_language]} source into natural spoken {LANGUAGES[target_language]} "
            "for a video dubbing slot. Adjust the wording without summarizing or dropping any clause. "
            "Preserve every statement, question, negation, name, number and essential fact. "
            "Return ONLY the dialogue without explanations.\n"
            f"The available speaking time is {seconds:.2f} seconds.\n"
            f"Aim for about {max(.01, seconds * .9):.2f} to {seconds:.2f} seconds of speech, "
            "not a much shorter line. If preserving every clause needs more time, keep the meaning; "
            "the audio can be sped up later.\n"
            f"Previous translation: {record['target_text']}\n"
            f"Previous natural audio: {record['tts_natural_duration']:.2f} seconds; "
            f"after native speed control: {record['tts_duration']:.2f} seconds.\n"
        )
        if record["tts_duration"] < .9 * seconds:
            prompt += "The previous dub was too short. Restore omitted clauses and detail; use longer natural wording.\n"
        else:
            prompt += "The previous dub was too long. Tighten wording while retaining every clause.\n"
        if target_language not in ("zh", "ja", "ko"):
            count = record["shortening_word_budget"]
            unit = "word" if count == 1 else "words"
            prompt += f"Aim for about {count} {LANGUAGES[target_language]} {unit}; do not cut meaning to hit this estimate.\n"
        else:
            prompt += (f"Aim for about {record['shortening_word_budget']} {LANGUAGES[target_language]} "
                       "characters; do not cut meaning to hit this estimate.\n")
        return prompt + f"\n[Source Text]\n{record['text']}"
    prompt = continuity + (
        f"Please translate the following text from {LANGUAGES[source_language]} into {LANGUAGES[target_language]}, "
        "taking the provided background information into consideration. "
        "Return ONLY the translated spoken words, without explanations or labels. "
        "Translate every clause, including introductory phrases and quotation cues; do not omit or summarize. "
        "Preserve negation, numbers, names, pronouns and every essential fact. "
        "A source fragment must remain a fragment. Keep conditional clauses conditional; "
        "do not add a question or conclusion from a neighboring line. "
        "Use natural spoken dialogue and spell numbers as spoken words, not digits. "
        "Do not shorten for timing in this first translation; audio timing is handled separately.\n"
    )
    return prompt + f"\n[Source Text]\n{record['text']}"


def validate_qwen_assets(model_dir):
    for name in QWEN_TTS_FILES:
        if not (Path(model_dir) / name).is_file():
            raise FileNotFoundError(Path(model_dir) / name)


def load_qwen(model_dir, device):
    import torch

    model_dir = Path(model_dir).resolve()
    validate_qwen_assets(model_dir)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MODELSCOPE_OFFLINE="1")
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return get(("qwen3tts", str(model_dir), device, str(dtype), "sdpa"),
               lambda: Qwen3TTSModel.from_pretrained(
                   str(model_dir), device_map=device, dtype=dtype, attn_implementation="sdpa"))


class QwenSpeechModel:
    """Local reference-conditioned Qwen3-TTS in its dedicated interpreter."""

    def __init__(self, config):
        self.config = config
        self.name = str(Path(config.tts_model).resolve())
        self.reference_mode = "x_vector_only" if config.qwen_x_vector_only_mode else "in_context"
        self.identity = {"backend": "qwen3tts", "model": self.name,
                         "files": [signature(Path(self.name) / name) for name in QWEN_TTS_FILES],
                         "reference_mode": self.reference_mode, "device": config.device}

    def generate(self, request_path):
        call_function(self.config.tts_python, "stages.speech.model", "qwen_worker",
                      {"request_path": str(request_path)},
                      env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, check=True)


class APISpeechModel:
    """Remote preset-voice synthesis; reference audio is used only for loudness."""

    def __init__(self, config):
        self.config = config
        self.name = config.tts_model
        self.reference_mode = "preset_voice"
        self.identity = {"backend": "openai_compatible", "model": self.name,
                         "url": config.tts_url.rstrip("/"), "voice": config.tts_voice,
                         "api_key_env": config.tts_api_key_env}

    def generate(self, request_path):
        request = json.loads(Path(request_path).read_text())
        client = SpeechSynthesizer(self.config.tts_url, self.name, self.config.tts_voice,
                                   self.config.tts_api_key_env)
        try:
            for row in request["rows"]:
                output = Path(row["path"])
                if output.is_file():
                    continue
                audio, rate = sf.read(io.BytesIO(client(row["text"])), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError("TTS API returned invalid audio")
                temporary = output.with_suffix(".tmp.wav")
                sf.write(temporary, audio, rate, subtype="FLOAT")
                temporary.replace(output)
        finally:
            client.close()


def create_speech_model(config):
    backends = {"qwen3tts": QwenSpeechModel, "openai_compatible": APISpeechModel}
    return backends[config.tts_backend](config)


def qwen_worker(request_path):
    import torch

    request = json.loads(Path(request_path).read_text())
    pending = [row for row in request["rows"] if not Path(row["path"]).is_file()]
    if not pending:
        return
    model = load_qwen(request["model"], request["device"])
    for row in pending:
        random.seed(row["seed"])
        np.random.seed(row["seed"])
        torch.manual_seed(row["seed"])
        audio, rate = model.generate_voice_clone(
            text=row["text"], language=LANGUAGES[request["language"]],
            ref_audio=row["reference"], ref_text=row.get("reference_text", ""),
            x_vector_only_mode=row.get("x_vector_only_mode", True),
        )
        values = np.asarray(audio[0], dtype=np.float32)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("Qwen3-TTS returned invalid audio")
        output = Path(row["path"])
        temporary = output.with_suffix(".tmp.wav")
        sf.write(temporary, values, rate, subtype="FLOAT")
        temporary.replace(output)
