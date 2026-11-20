"""Verify acoustic MOSS split boundaries by re-recognizing each side with MOSS.

The acoustic stage (mora prior + energy valley) proposes boundaries. This
module asks MOSS to transcribe the audio on both sides of each boundary. A
boundary is accepted when both sides are re-recognized as their expected
sentences. Otherwise it is recomputed: candidate boundaries are taken from
neighbouring energy valleys, ordered by the direction implied by the
recognition mismatch, plus a character-rate estimate, and each candidate is
verified the same way.

Verification transcripts only judge timing. Sentence text, speakers, and
lineage always come from the first-pass MOSS blocks. Inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata
from typing import Callable, Protocol, Sequence

from ..analysis_plugins import AnalysisPluginError


class TranscribePort(Protocol):
    def request(self, route: str, audio_path: str, data: dict, *,
                start_ms: int, end_ms: int) -> tuple[dict, int]: ...


class VerificationUnavailable(Exception):
    """MOSS could not be reached or returned an unusable verification body."""


def _lexical(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum())


def similarity(expected: str, recognized: str, *, prefix: bool = False) -> float:
    """Lexical similarity in [0, 1]; prefix mode ignores trailing extra speech."""
    want, got = _lexical(expected), _lexical(recognized)
    if prefix:
        got = got[:len(want)]
    if not want and not got:
        return 1.0
    return SequenceMatcher(None, want, got, autojunk=False).ratio()


def energy_valleys(audio_path: str, lo_ms: int, hi_ms: int, *,
                   window_ms: int = 20, step_ms: int = 10, limit: int = 8) -> list[int]:
    """Return local RMS minima inside [lo_ms, hi_ms], deepest first.

    Unreadable or non-PCM16 audio yields no candidates; callers fall back to
    a character-rate estimate.
    """
    try:
        import struct
        import wave
        with wave.open(audio_path, "rb") as wf:
            rate, channels = wf.getframerate(), wf.getnchannels()
            if wf.getsampwidth() != 2 or hi_ms - lo_ms <= window_ms:
                return []
            wf.setpos(int(lo_ms * rate / 1000))
            raw = wf.readframes(int((hi_ms - lo_ms) * rate / 1000))
        total = len(raw) // (2 * channels)
        win, step = int(window_ms * rate / 1000), max(1, int(step_ms * rate / 1000))
        if total <= win or win <= 0:
            return []
        data = struct.unpack(f"<{total * channels}h", raw[:total * channels * 2])
        mono = data[::channels]
        profile = []
        for offset in range(0, total - win, step):
            window = mono[offset:offset + win]
            profile.append((lo_ms + int((offset + win // 2) * 1000 / rate),
                            sum(s * s for s in window) / win))
        minima = [profile[i] for i in range(1, len(profile) - 1)
                  if profile[i][1] <= profile[i - 1][1] and profile[i][1] <= profile[i + 1][1]]
        minima.sort(key=lambda item: item[1])
        selected: list[int] = []
        for position, _ in minima:
            if all(abs(position - other) >= 60 for other in selected):
                selected.append(position)
            if len(selected) >= limit:
                break
        return selected
    except Exception:
        return []


@dataclass(frozen=True)
class VerificationPolicy:
    min_verify_duration_ms: int = 800
    min_sentence_duration_ms: int = 280
    similarity_threshold: float = 0.8
    max_candidates: int = 3


class MossBoundaryVerifier:
    """Check each internal boundary with MOSS and recompute rejected ones."""

    def __init__(self, port: TranscribePort, audio_path: str, policy: VerificationPolicy, *,
                 valley_finder: Callable[[str, int, int], list[int]] = energy_valleys):
        self.port, self.audio_path, self.policy = port, audio_path, policy
        self.valley_finder = valley_finder
        self._cache: dict[tuple[int, int], str] = {}

    def _recognize(self, start_ms: int, end_ms: int) -> str:
        key = (start_ms, end_ms)
        if key not in self._cache:
            try:
                body, _ = self.port.request("/transcribe", self.audio_path, {"language": "ja"},
                                            start_ms=start_ms, end_ms=end_ms)
            except AnalysisPluginError as exc:
                raise VerificationUnavailable(getattr(exc, "code", None) or str(exc)) from exc
            blocks = body.get("blocks") if isinstance(body, dict) else None
            if not isinstance(blocks, list):
                raise VerificationUnavailable("invalid_verification_body")
            self._cache[key] = "".join(
                b.get("text", "") for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str))
        return self._cache[key]

    def _evaluate(self, lo: int, boundary: int, hi: int, left: str, right: str,
                  right_is_final: bool) -> dict:
        left_rec = self._recognize(lo, boundary)
        right_rec = self._recognize(boundary, hi)
        left_score = similarity(left, left_rec)
        right_score = similarity(right, right_rec, prefix=not right_is_final)
        threshold = self.policy.similarity_threshold
        return {"boundaryMs": boundary, "leftText": left_rec, "rightText": right_rec,
                "leftScore": round(left_score, 4), "rightScore": round(right_score, 4),
                "passed": left_score >= threshold and right_score >= threshold,
                "score": left_score + right_score}

    def _candidates(self, lo: int, hi: int, original: int, left: str, right: str,
                    left_rec: str, tried: set[int]) -> list[int]:
        floor = lo + self.policy.min_sentence_duration_ms
        ceiling = hi - self.policy.min_sentence_duration_ms
        # Positive: left side lost characters, so the boundary is too early.
        missing = len(_lexical(left)) - len(_lexical(left_rec))
        direction = (missing > 0) - (missing < 0)
        valleys = [v for v in self.valley_finder(self.audio_path, floor, ceiling)
                   if floor <= v <= ceiling]
        preferred = [v for v in valleys if direction == 0 or (v - original) * direction > 0]
        others = [v for v in valleys if v not in preferred]
        ordered = preferred + others
        total_chars = max(1, len(_lexical(left)) + len(_lexical(right)))
        estimate = original + round(missing * (hi - lo) / total_chars)
        estimate = max(floor, min(estimate, ceiling))
        if direction:
            ordered.insert(min(1, len(ordered)), estimate)
        result: list[int] = []
        for candidate in ordered:
            if all(abs(candidate - seen) >= 40 for seen in tried | set(result)):
                result.append(candidate)
        return result[:self.policy.max_candidates]

    def verify(self, parts: Sequence[str], edges: Sequence[int]) -> tuple[tuple[int, ...], tuple[dict, ...]]:
        """Return verified edges and one audit record per internal boundary.

        Raises VerificationUnavailable when MOSS cannot be queried; callers
        keep the acoustic boundaries in that case.
        """
        current = list(edges)
        audits = []
        last = len(current) - 2
        for k in range(1, len(current) - 1):
            lo, original, hi = current[k - 1], current[k], current[k + 1]
            left, right = parts[k - 1], parts[k]
            if (original - lo < self.policy.min_verify_duration_ms
                    or hi - original < self.policy.min_verify_duration_ms):
                audits.append({"boundary": k, "originalMs": original, "finalMs": original,
                               "status": "skipped", "attempts": []})
                continue
            first = self._evaluate(lo, original, hi, left, right, k == last)
            attempts, best, status = [first], first, "verified"
            if not first["passed"]:
                status = "unverified"
                tried = {original}
                for candidate in self._candidates(lo, hi, original, left, right,
                                                  first["leftText"], tried):
                    tried.add(candidate)
                    attempt = self._evaluate(lo, candidate, hi, left, right, k == last)
                    attempts.append(attempt)
                    if attempt["passed"]:
                        best, status = attempt, "recomputed"
                        break
                    if attempt["score"] > best["score"]:
                        best, status = attempt, "improved"
            current[k] = best["boundaryMs"]
            audits.append({"boundary": k, "originalMs": original, "finalMs": current[k],
                           "status": status, "attempts": attempts})
        return tuple(current), tuple(audits)
