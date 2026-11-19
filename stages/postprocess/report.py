"""汇总实际音频替换和口型结果，不要求人工验收才能结束。"""

import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def report(run):
    state = read(run / "run.json")
    settings = state["request"]["config"]
    mix = read(run / "audio/audio_mix.json")
    rendered_jobs = read(run / "render/render_manifest.json")["jobs"]
    lip_results = {}
    for job in rendered_jobs:
        for item in job.get("utterance_ids", [job["utterance_id"]]):
            lip_results.setdefault(item, []).append(job["status"])
    video_reasons = {item: job.get("reason", "") for job in rendered_jobs if job["status"] != "complete"
                     for item in job.get("utterance_ids", [job["utterance_id"]])}
    checked = Path(state.get("selected_dialogue_path", run / "speech/dialogue.json"))
    rows = []
    dialogue = read(checked)
    accepted = set(mix["metadata"]["accepted_replacement_ids"])
    rejected = {row["id"]: row["reason"] for row in mix["metadata"]["skipped_replacements"]}
    for row in dialogue["utterances"]:
        translated = row["id"] in accepted
        statuses = lip_results.get(row["id"], [])
        video = "preserve_original"
        if translated and statuses:
            if all(status == "complete" for status in statuses):
                video = "lip_edited"
            elif "complete" in statuses:
                video = "partially_lip_edited"
        rows.append({"id": row["id"], "source_text": row["text"], "target_text": row.get("target_text", ""),
                     "translatable_speech": row.get("human_speech", True),
                     "audio": "translated" if translated else "preserve_original",
                     "video": video,
                     "video_reason": video_reasons.get(row["id"], ""),
                     "reason": "" if translated else rejected.get(row["id"], row.get("reason", "not_accepted"))})
        if row.get("source_parts"):
            rows[-1]["source_utterance_ids"] = [part["id"] for part in row["source_parts"]]
        for key in ("max_tempo", "time_compression", "translation_review"):
            if key in row:
                rows[-1][key] = row[key]
    (run / "source.txt").write_text("\n".join(f"{row['id']}: {row['source_text']}" for row in rows) + "\n")
    (run / "translated.txt").write_text("\n".join(
        f"{row['id']}: {row['target_text'] if row['audio'] == 'translated' else '[original audio preserved]'}"
        for row in rows) + "\n")
    result = {"status": "complete", "automatic": True, "human_review_required": False,
              "source_language": dialogue.get("source_language", settings["source_language"]),
              "target_language": settings["target_language"],
              "profile": "modern",
              "duration_mode": settings.get("duration_mode", "fixed"),
              "video": str(run / "final.mp4"), "utterances": rows,
              "translated": sum(row["audio"] == "translated" for row in rows),
              "preserved_original": sum(row["audio"] == "preserve_original" for row in rows),
              "lip_edited": sum(row["video"] == "lip_edited" for row in rows),
              "partial_lip_edited": sum(row["video"] == "partially_lip_edited" for row in rows),
              "audio_only": sum(row["audio"] == "translated" and row["video"] == "preserve_original" for row in rows),
              "audio_policy": mix["metadata"].get("source_background_method", "external_target_audio"),
              "limitations": ["Speech recognition, translation and separation are model estimates.",
                              "Non-speech mixed with dialogue may be affected by separation errors."]}
    if "timing_fit" in dialogue:
        result["timing_fit"] = dialogue["timing_fit"]
    if "timings" in dialogue:
        result["speech_timings"] = dialogue["timings"]
    result["content_verification"] = "disabled"
    result["limitations"].append("The report records translation and timing checks. It does not independently transcribe the generated soundtrack.")
    edited = result["lip_edited"] + result["partial_lip_edited"]
    if not result["translated"]:
        result["video_mode"] = "preserve_original"
    elif not edited:
        result["video_mode"] = "audio_only"
    elif result["partial_lip_edited"] or result["audio_only"]:
        result["video_mode"] = "partially_lip_edited"
    else:
        result["video_mode"] = "lip_edited"
    spoken = [row for row in rows if row["translatable_speech"]]
    if settings.get("merge_speaker_turns", False):
        result["speaker_turn_merging"] = {**settings["speaker_turn_merge_policy"],
            "source_utterances": sum(len(row.get("source_utterance_ids", [row["id"]])) for row in spoken),
            "translation_units": len(spoken)}
    result["audio_completeness"] = {"recognized_dialogue": len(spoken),
        "replaced_dialogue": sum(row["audio"] == "translated" for row in spoken),
        "unreplaced_ids": [row["id"] for row in spoken if row["audio"] != "translated"],
        "scope": "recognized_dialogue_only_not_a_guarantee_of_zero_asr_omissions"}
    if dialogue.get("audio_policy") == "complete_dialogue":
        result["delivery_policy"] = "complete_dialogue"
        result["limitations"].append("The selected separator's full background is estimated and may change non-speech sounds or retain faint speech.")
    if "background_balance" in mix["metadata"]:
        result["background_balance"] = mix["metadata"]["background_balance"]
    if any(row.get("translation_review") == "disabled" for row in rows):
        result.update(translation_review="disabled", time_compression_policy="uncapped", max_tempo=None)
        result["limitations"][0] = "Translation is not independently semantically reviewed; synthesized speech is not checked by ASR."
        result["limitations"].append("Uncapped time compression can reduce speech naturalness and intelligibility.")
    temporary = run / "report.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(run / "report.json")
    print(json.dumps({key: result[key] for key in ("translated", "preserved_original", "lip_edited", "partial_lip_edited", "audio_only")}))


def run_stage(*, run_dir):
    return report(Path(run_dir).resolve())
