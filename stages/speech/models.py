"""Translation prompts and the Qwen3-TTS voice-cloning worker."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from runtime.languages import LANGUAGES
from runtime.models import get

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


def prepare_waveform(audio, rate, reference_rms):
    """Keep short guards around actual speech, then match the reference's RMS."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("invalid_tts_audio")
    block = max(1, round(rate * .02))
    energy = np.array([np.sqrt(np.mean(audio[index:index + block] ** 2))
                       for index in range(0, len(audio), block)])
    active = np.flatnonzero(energy > max(1e-4, float(energy.max()) * .015))
    if not active.size:
        raise ValueError("silent_tts_audio")
    first = max(0, int(active[0]) * block - round(.04 * rate))
    last = min(len(audio), (int(active[-1]) + 1) * block + round(.06 * rate))
    selected = audio[first:last]
    rms = float(np.sqrt(np.mean(selected ** 2)))
    gain = float(np.clip(reference_rms / max(rms, 1e-8), .5, 2.0)) if reference_rms else 1.0
    peak = float(np.max(np.abs(selected)))
    gain = min(gain, .98 / peak) if peak else gain
    selected = selected * gain
    # Some codecs end the generated utterance without a release guard. Touch
    # only the final 8 ms, then add the missing guard rather than trimming speech.
    active_end = min(len(audio), (int(active[-1]) + 1) * block)
    padding = max(0, round(.04 * rate) - (last - active_end))
    fade = min(len(selected), max(1, round(.008 * rate))) if padding else 0
    if fade:
        selected[-fade:] *= (.5 + .5 * np.cos(np.linspace(0, np.pi, fade))).astype(np.float32)
        selected[-1] = 0
        selected = np.pad(selected, (0, padding))
    return selected, {"trimmed_leading_seconds": first / rate,
                             "trimmed_trailing_seconds": (len(audio) - last) / rate,
                             "rms_before_gain": rms, "speech_gain": gain,
                             "duration_before_trim": len(audio) / rate,
                             "tail_fade_seconds": fade / rate, "padded_trailing_seconds": padding / rate}


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


def qwen_worker(request_path):
    import torch

    request = json.loads(Path(request_path).read_text())
    pending = [row for row in request["rows"] if not Path(row["path"]).is_file()]
    if not pending:
        return
    model = load_qwen(request["model_dir"], request["device"])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-request", required=True)
    qwen_worker(parser.parse_args().qwen_request)
