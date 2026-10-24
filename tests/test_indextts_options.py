import unittest

from modules.tts.indextts_options import IndexTTSOptions


class IndexTTSOptionsTest(unittest.TestCase):
    def test_partial_options_keep_generation_defaults(self):
        options = IndexTTSOptions.from_dict({"emotion_mode": "text", "emotion_text": "温柔"})
        self.assertEqual(options.max_text_tokens_per_segment, 120)
        self.assertEqual(options.emotion_text, "温柔")

    def test_invalid_configuration_is_rejected(self):
        cases = [
            {"emotion_mode": "invalid"}, {"language": "FR"},
            {"duration_factor": float("nan")}, {"temperature": float("inf")},
            {"top_k": 1.5}, {"num_beams": 0}, {"max_mel_tokens": 49},
            {"max_text_tokens_per_segment": 19}, {"do_sample": "false"},
            {"emotion_vector": [0] * 7}, {"emotion_vector": [float("nan")] * 8},
            {"emotion_mode": "reference"}, {"unsupported": True},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                IndexTTSOptions.from_dict(values)
