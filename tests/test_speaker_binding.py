"""Regressions for audio ownership across scene cuts."""

import unittest

from stages.lipsync.timeline import plan_jobs
from stages.speakers.bind import build_plan


def segment(speaker, words):
    return {"speaker_id": speaker, "speaker_ids": [speaker], "start": words[0][1], "end": words[-1][2],
            "text": " ".join(text for text, _, _ in words),
            "words": [{"speaker_id": speaker, "speaker_ids": [speaker], "text": text, "start": start, "end": end}
                      for text, start, end in words]}


def analyzed_plan(segments):
    def track(shot, start, end, score):
        count = end - start
        return {"id": shot, "shot_id": shot, "frames": list(range(start, end)),
                "asd_scores": [score] * count, "bboxes": [[0, 0, 16, 16]] * count,
                "confidence": [.9] * count}

    shots = [{"id": "left", "start_frame": 0, "end_frame": 25},
             {"id": "right", "start_frame": 25, "end_frame": 50}]
    vision = {"fps": 25, "frames": 50, "width": 32, "height": 32, "video_path": "sample.mp4",
              "shots": shots, "tracks": [track("left", 0, 25, 3.), track("right", 25, 50, 2.)], "models": {}}
    identity = {"track_to_person": {"left": "alice", "right": "bob"}, "models": {},
                "tracks": [{"id": name, "quality": {"match_eligible": True}} for name in ("left", "right")]}
    audio = {"source_audio": "sample.wav", "segments": segments, "models": {"diarization": {"backend": "test"}}}
    return build_plan(vision, identity, [audio], merge_turns=True)


class SpeakerBindingTests(unittest.TestCase):
    def test_reaction_cut_keeps_words_with_their_voice(self):
        plan = analyzed_plan(
            [segment("a", [("Going", .40, .64), ("home", .64, 1.), ("now?", 1., 1.16)]),
             segment("b", [("Yes.", 1.28, 1.56)])]
        )
        question, reply = plan["utterances"]
        self.assertEqual((question["text"], question["end"], question["person_id"]),
                         ("Going home now?", 1.16, "alice"))
        self.assertEqual((reply["text"], reply["start"], reply["person_id"]), ("Yes.", 1.28, "bob"))
        self.assertEqual((question["acoustic_speaker_ids"], reply["acoustic_speaker_ids"]), (["a"], ["b"]))

        target = {"start": question["start"], "end": question["end"], "text": "这就回家？"}
        jobs = plan_jobs(plan, [target], [{"source_index": 0, "target_indices": [0]}], .1, .1)
        edits = [job for job in jobs if job["action"] == "tbdub"]
        self.assertEqual([(job["person_id"], job["shot_id"], job["end_frame"]) for job in edits],
                         [("alice", "left", 25)])

    def test_same_face_cannot_merge_different_acoustic_speakers(self):
        plan = analyzed_plan([segment("a", [("Hello.", 1.08, 1.32)]),
                              segment("b", [("Hi.", 1.40, 1.64)])])
        self.assertEqual([row["text"] for row in plan["utterances"]], ["Hello.", "Hi."])
