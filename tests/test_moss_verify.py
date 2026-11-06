import unittest
from unittest.mock import Mock

from modules.analysis_plugins import AnalysisPluginError
from modules.detect.moss_verify import (MossBoundaryVerifier, VerificationPolicy,
                                        VerificationUnavailable, energy_valleys, similarity)


def _transcriber(pause_ms: int):
    """MOSS stub whose true sentence pause is at pause_ms inside 2000-6000 ms."""
    def transcribe(route, path, data, start_ms, end_ms):
        if start_ms == 2000:
            text = "こんにちは" if end_ms >= pause_ms else "こんに"
            if end_ms > pause_ms + 400:
                text += "は"
        else:
            text = "はい" if start_ms >= pause_ms else "ちははい"
        return {"blocks": [{"text": text}]}, end_ms - start_ms
    return transcribe


class SimilarityTests(unittest.TestCase):
    def test_ignores_punctuation_and_width(self):
        self.assertEqual(similarity("こんにちは。", "こんにちは"), 1.0)

    def test_prefix_mode_ignores_trailing_speech(self):
        self.assertEqual(similarity("はい。", "はいそうです", prefix=True), 1.0)
        self.assertLess(similarity("はい。", "はいそうです"), 0.8)


class MossBoundaryVerifierTests(unittest.TestCase):
    def test_rejected_boundary_is_recomputed_from_neighbouring_valleys(self):
        port = Mock()
        port.request.side_effect = _transcriber(3600)
        verifier = MossBoundaryVerifier(
            port, "in.wav", VerificationPolicy(),
            valley_finder=lambda path, lo, hi: [2700, 3600, 5000])
        edges, audits = verifier.verify(("こんにちは。", "はい。"), (2000, 3000, 6000))
        self.assertEqual(edges, (2000, 3600, 6000))
        self.assertEqual(audits[0]["status"], "recomputed")
        self.assertEqual(audits[0]["attempts"][-1]["boundaryMs"], 3600)
        # The mismatch says "too early", so earlier valleys are not tried first.
        self.assertNotEqual(audits[0]["attempts"][1]["boundaryMs"], 2700)

    def test_unresolved_boundary_keeps_best_attempt_and_stays_ordered(self):
        port = Mock()
        port.request.side_effect = lambda *a, start_ms, end_ms, **k: (
            {"blocks": [{"text": "ざつおん"}]}, end_ms - start_ms)
        verifier = MossBoundaryVerifier(
            port, "in.wav", VerificationPolicy(max_candidates=2),
            valley_finder=lambda path, lo, hi: [3300, 4200])
        edges, audits = verifier.verify(("こんにちは。", "はい。"), (2000, 3000, 6000))
        self.assertEqual(audits[0]["status"], "unverified")
        self.assertEqual(edges, (2000, 3000, 6000))
        self.assertEqual(len(audits[0]["attempts"]), 3)

    def test_short_sides_are_not_sent_to_moss(self):
        port = Mock()
        verifier = MossBoundaryVerifier(port, "in.wav", VerificationPolicy(min_verify_duration_ms=800))
        edges, audits = verifier.verify(("はい。", "うん。"), (0, 500, 1200))
        self.assertEqual(edges, (0, 500, 1200))
        self.assertEqual(audits[0]["status"], "skipped")
        port.request.assert_not_called()

    def test_transport_failure_raises_typed_error(self):
        port = Mock()
        port.request.side_effect = AnalysisPluginError("502 Bad Gateway")
        verifier = MossBoundaryVerifier(port, "in.wav", VerificationPolicy(),
                                        valley_finder=lambda *a: [])
        with self.assertRaises(VerificationUnavailable):
            verifier.verify(("こんにちは。", "はい。"), (2000, 3000, 6000))

    def test_inputs_are_not_mutated(self):
        port = Mock()
        port.request.side_effect = _transcriber(3600)
        edges = [2000, 3000, 6000]
        MossBoundaryVerifier(port, "in.wav", VerificationPolicy(),
                             valley_finder=lambda *a: [3600]).verify(("こんにちは。", "はい。"), edges)
        self.assertEqual(edges, [2000, 3000, 6000])


class EnergyValleyTests(unittest.TestCase):
    def test_unreadable_audio_yields_no_candidates(self):
        self.assertEqual(energy_valleys("missing.wav", 0, 1000), [])


if __name__ == "__main__":
    unittest.main()
