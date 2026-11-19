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
from stages.speech.timing import CJK, available_seconds, text_units, unit_budget

QWEN_TTS_FILES = ["config.json", "model.safetensors", "speech_tokenizer/config.json",
                  "speech_tokenizer/model.safetensors"]


REVIEW_SYSTEM = (
    "Check source-line facts in a video dub. Output one line only: PASS or "
    "FAIL: <one specific factual mismatch and a correction, under 25 words>. Do not analyze out loud."
)


def dialogue_context(record, conversation):
    position = next((index for index, item in enumerate(conversation)
                     if item["id"] == record["id"]), len(conversation))
    neighbors = conversation[max(0, position - 8):position + 9]
    previous = next((item for item in reversed(conversation[:position])
                     if item["speaker_id"] == record["speaker_id"]), None)
    if previous is not None and previous not in neighbors:
        neighbors = [previous, *neighbors]
    return "\n".join(f"{item['id']} ({item['speaker_id']}): {item['text'][:160]}"
                     for item in neighbors if item["id"] != record["id"])


def translation_prompt(record, conversation, source_language, target_language, shorten=False):
    seconds = available_seconds(record)
    budget = record.get("shortening_word_budget", unit_budget(seconds, target_language))
    unit = "spoken characters" if target_language in CJK else "spoken words"
    context = dialogue_context(record, conversation)
    source = f"\n[Context, do not translate]\n{context}\n\n[Source line]\n{record['text']}"
    instructions = (
        f"Translate the source line from {LANGUAGES[source_language]} into idiomatic, spoken "
        f"{LANGUAGES[target_language]} for a video dub. Return only the spoken translation.\n"
        "Keep the speaker's tone and every essential fact, named entity, number, negation, and question. "
        "Use the conversation to resolve ambiguous words, names, and what pronouns refer to. "
        "Render the meaning in the most concise natural wording. Remove filler and literal phrasing "
        "where the meaning and speaking intent stay intact. Never borrow words from neighboring speakers. "
        "Keep deliberate repetitions that matter for the delivery. Spell numbers as they are spoken.\n"
        f"The available time is {seconds:.2f} seconds. Aim for at most about {budget} {unit} "
        "at natural speed. A brief standalone reaction can use its shortest idiomatic equivalent. "
        "Keep the meaning if this estimate is too tight.\n"
    )
    if shorten:
        previous = record["target_text"]
        previous_units = text_units(previous, target_language)
        instructions += (
            f"The previous version took {record['tts_duration']:.2f} seconds "
            f"and contains {previous_units} {unit}: {previous}\n"
            f"Rewrite it with different wording, aiming for {budget} {unit}. "
            "Favor direct everyday phrasing, short expressions of negation and implied words "
            "where clear. Keep the same meaning and use delivery rather than extra words for emphasis. "
            "Repeating the previous wording cannot fit the time budget.\n"
        )
    if record.get("try_different_wording"):
        instructions += (
            "Your last attempt repeated the version that ran too long. "
            "Use a different grammatical construction to express the same idea in fewer spoken words.\n"
        )
    if record.get("rejected_translation"):
        instructions += (
            f"Rejected translation: {record['rejected_translation']}\n"
            f"Review feedback: {record['review_feedback']}\n"
            "Correct the flagged mismatch using the source and context. Preserve every essential fact.\n"
        )
    return instructions + source


def semantic_review_prompt(record, target, source_language, target_language):
    return (
        f"Original ({LANGUAGES[source_language]}): {record['text']}\n"
        f"Previous dub ({LANGUAGES[target_language]}): {record['target_text']}\n"
        f"Shorter dub for review ({LANGUAGES[target_language]}): {target}\n"
        "Judge whether the shorter dub preserves the original line's facts and intent. "
        "Natural grammar, brief idiomatic reactions, and omitted filler are allowed. "
        "Do not invent facts or reject a stylistic change. If there is no clear error, answer PASS. "
        "Otherwise answer FAIL: followed by the concrete change and a short correction."
    )


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
