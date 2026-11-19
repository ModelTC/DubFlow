"""Regressions for repairing coarse forced-alignment timestamps."""
import unittest
from unittest.mock import Mock

from stages.recognition.alignment import repair_word_times
from stages.recognition.utils import SAMPLE_RATE, label_words, sentence_segments


def words(*items):
    return [{"word": text, "start": start, "end": end} for text, start, end in items]


class AlignmentRepairTests(unittest.TestCase):
    def test_repair_recovers_reactions_and_keeps_sentence_tail_with_its_speaker(self):
        transcript = "Before. What? What? What? I found it. After."
        raw = words(("Before", 16.32, 16.48), ("What", 17.52, 17.60),
                    ("What", 17.60, 17.60), ("What", 17.60, 17.60),
                    ("I", 17.60, 17.60), ("found", 18.40, 18.40),
                    ("it", 19.68, 19.84), ("After", 21.2, 21.44))
        turns = [{"start": 16.93, "end": 19.64, "speaker_id": "voice"}]
        realign = Mock(return_value=words(
            ("What", 0., 0.), ("What", .72, .8), ("What", .8, 1.12),
            ("I", 1.12, 1.44), ("found", 1.60, 1.68), ("it", 2.96, 3.04)))
        recovered, repairs = repair_word_times(raw, transcript, bytearray(23 * SAMPLE_RATE), "en", realign, turns)
        self.assertEqual(len(repairs), 1)
        realign.assert_called_once()
        self.assertEqual(realign.call_args[0][1].strip(), "What? What? What? I found it.")
        self.assertEqual(repairs[0]["missing_characters_before"], 14)
        self.assertEqual(repairs[0]["missing_characters_after"], 0)
        self.assertEqual(recovered[0]["start"], raw[0]["start"])
        self.assertEqual(recovered[-1]["end"], raw[-1]["end"])
        segments = sentence_segments(recovered, transcript)
        labeled = label_words({"aligned_segments": segments}, turns)
        self.assertEqual([row["text"] for row in labeled], ["Before.", "What? What? What?", "I found it.", "After."])
        self.assertEqual(labeled[2]["speaker_ids"], ["voice"])
        self.assertEqual(labeled[2]["words"][-1]["speaker_assignment"], "recovered_turn_boundary")

    def test_brief_positive_reaction_and_one_untimed_function_word_keep_their_times(self):
        transcript = "What? I want to go."
        raw = words(("What", .16, .32), ("I", .64, .72), ("want", .72, .88), ("to", .88, .88), ("go", .88, 1.12))
        realign = Mock()
        recovered, repairs = repair_word_times(raw, transcript, bytearray(2 * SAMPLE_RATE), "en", realign)
        realign.assert_not_called()
        self.assertEqual(repairs, [])
        self.assertEqual([(row["start"], row["end"]) for row in recovered],
                         [(row["start"], row["end"]) for row in raw])


if __name__ == "__main__":
    unittest.main()
