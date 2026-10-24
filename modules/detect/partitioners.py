"""Joint dynamic programming over Japanese token and fine-VAD boundaries."""

from bisect import bisect_left
from math import log

from .contracts import (
    DetectConfig,
    DetectError,
    JointPartition,
    JointSegment,
    RegionPartitioner,
)


_HARD_CONTINUATIONS = frozenset({
    "relative_clause",
    "nominal_continuation",
    "continuing_particle_follows",
    "terminal_particle_follows",
    "suspect_asr_word_fragment",
    "fixed_expression_continuation",
})


def _acoustic_score(gap_ms: int) -> float:
    """A 200ms gap is evidence; gaps near 700ms are independently strong."""
    if gap_ms <= 0:
        return 0.0
    return min(1.0, (gap_ms / 700.0) ** 0.65)


def _nearest_prefix(prefix, target, upper):
    index = bisect_left(prefix, target, 1, upper)
    choices = {min(max(index, 1), upper - 1)}
    if index > 1:
        choices.add(index - 1)
    return min(choices, key=lambda item: (abs(prefix[item] - target), item))


class JointDynamicProgrammingPartitioner:
    """Infer sentence count, Japanese cuts, and atom ownership together.

    Strong morphology boundaries are always searched. To tolerate missing or
    corrupted ASR punctuation, every acoustic boundary also projects onto the
    nearest reading-length position in the Japanese text. Large candidate sets
    retain the strongest gaps and their language projections, bounding runtime
    without reverting to greedy decisions.
    """

    _MAX_TEXT_CANDIDATES = 48
    _MAX_AUDIO_CANDIDATES = 40

    def __init__(self, config: DetectConfig):
        self.config = config

    def _candidate_ends(self, detection, regions, token_prefix, voiced_prefix):
        token_count, atom_count = len(detection.tokens), len(regions)
        boundary_by_token = {row.token_index: row for row in detection.boundaries}
        if set(boundary_by_token) != set(range(1, token_count + 1)):
            raise DetectError(
                "DETECT_ANALYSIS_INVALID",
                "Sentence detector must describe every token boundary",
            )

        linguistic = {
            row.token_index
            for row in detection.boundaries[:-1]
            if row.language_score >= 0.22 or any(
                reason in row.reasons
                for reason in (
                    "terminal_punctuation",
                    "quoted_terminal",
                    "terminal_particle",
                    "new_sentence_connector",
                    "new_clause_starter",
                )
            )
        }
        projected_text = set()
        total_units, total_voiced = token_prefix[-1], voiced_prefix[-1]
        for atom_end in range(1, atom_count):
            target = total_units * voiced_prefix[atom_end] / total_voiced
            nearest = _nearest_prefix(token_prefix, target, token_count + 1)
            projected_text.update(
                index for index in (nearest - 1, nearest, nearest + 1)
                if 0 < index < token_count
            )
        text_ends = linguistic | projected_text
        if len(text_ends) > self._MAX_TEXT_CANDIDATES - 1:
            ranked = sorted(
                text_ends,
                key=lambda index: (
                    index in linguistic,
                    boundary_by_token[index].language_score,
                    index in projected_text,
                    -index,
                ),
                reverse=True,
            )
            text_ends = set(ranked[:self._MAX_TEXT_CANDIDATES - 1])
        text_ends.add(token_count)

        if atom_count <= self._MAX_AUDIO_CANDIDATES:
            audio_ends = set(range(1, atom_count + 1))
        else:
            gaps = {
                index: max(
                    0,
                    regions[index].speech_start_ms
                    - regions[index - 1].speech_end_ms,
                )
                for index in range(1, atom_count)
            }
            ranked_gaps = sorted(gaps, key=lambda index: (-gaps[index], index))
            audio_ends = set(ranked_gaps[:self._MAX_AUDIO_CANDIDATES // 2])
            for token_end in linguistic:
                target = total_voiced * token_prefix[token_end] / total_units
                nearest = _nearest_prefix(voiced_prefix, target, atom_count + 1)
                audio_ends.update(
                    index for index in (nearest - 1, nearest, nearest + 1)
                    if 0 < index < atom_count
                )
            if len(audio_ends) > self._MAX_AUDIO_CANDIDATES - 1:
                audio_ends = set(sorted(
                    audio_ends,
                    key=lambda index: (-gaps.get(index, 0), index),
                )[:self._MAX_AUDIO_CANDIDATES - 1])
            audio_ends.add(atom_count)
        return sorted(text_ends), sorted(audio_ends), boundary_by_token

    def partition(self, detection, regions, estimator):
        tokens = detection.tokens
        token_count, atom_count = len(tokens), len(regions)
        if not tokens or not regions:
            return None

        token_prefix = [0.0]
        for token in tokens:
            token_prefix.append(
                token_prefix[-1] + estimator.estimate_units((token,))
            )
        voiced_prefix = [0]
        for region in regions:
            voiced_prefix.append(
                voiced_prefix[-1] + region.speech_end_ms - region.speech_start_ms
            )
        total_units, total_voiced = token_prefix[-1], voiced_prefix[-1]
        if total_units <= 0 or total_voiced <= 0:
            return None

        text_ends, audio_ends, boundaries = self._candidate_ends(
            detection, regions, token_prefix, voiced_prefix
        )
        spoken_seconds = total_voiced / 1000.0
        global_rate = total_units / max(spoken_seconds, 0.001)
        rate_reliability = 1.0 if 2.5 <= global_rate <= 13.5 else 0.45
        duration_weight = self.config.duration_weight * (
            0.35 + 0.65 * detection.parse_confidence
        ) * rate_reliability

        # State: cost, segment count, predecessor, final segment. The graph is a
        # DAG because both token and atom indices strictly increase.
        best = {(0, 0): (0.0, 0, None, None)}
        for token_end in text_ends:
            candidate_audio = (
                (atom_count,) if token_end == token_count
                else [index for index in audio_ends if index < atom_count]
            )
            for atom_end in candidate_audio:
                winner = None
                for (token_start, atom_start), state in tuple(best.items()):
                    prior_cost, prior_count, _, _ = state
                    if (
                        token_start >= token_end
                        or atom_start >= atom_end
                        or prior_count >= self.config.max_sentence_count
                    ):
                        continue
                    units = token_prefix[token_end] - token_prefix[token_start]
                    observed = voiced_prefix[atom_end] - voiced_prefix[atom_start]
                    if units <= 0 or observed <= 0:
                        continue
                    expected = total_voiced * units / total_units
                    ratio = observed / expected
                    log_error = min(abs(log(max(ratio, 1e-9))), 1.6)
                    cost = prior_cost + duration_weight * log_error * log_error

                    span_ms = (
                        regions[atom_end - 1].speech_end_ms
                        - regions[atom_start].speech_start_ms
                    )
                    if units < self.config.min_sentence_units:
                        cost += 0.35 * (self.config.min_sentence_units - units)
                    if span_ms < self.config.min_sentence_duration_ms:
                        cost += (
                            self.config.min_sentence_duration_ms - span_ms
                        ) / self.config.min_sentence_duration_ms
                    if span_ms > self.config.max_sentence_duration_ms:
                        cost += 0.18 * (
                            span_ms / self.config.max_sentence_duration_ms - 1.0
                        )

                    boundary_score = 1.0
                    if token_end < token_count:
                        evidence = boundaries[token_end]
                        gap_ms = max(
                            0,
                            regions[atom_end].speech_start_ms
                            - regions[atom_end - 1].speech_end_ms,
                        )
                        acoustic = _acoustic_score(gap_ms)
                        language = max(
                            evidence.language_score,
                            self.config.unknown_language_floor
                            * (1.0 - detection.parse_confidence),
                        )
                        boundary_score = min(
                            1.5,
                            self.config.language_weight * language
                            + self.config.acoustic_weight * acoustic
                            + self.config.boundary_synergy_weight
                            * min(language, acoustic),
                        )
                        cost += self.config.boundary_bias - boundary_score
                        hard_continuation = bool(
                            _HARD_CONTINUATIONS.intersection(evidence.reasons)
                        ) or (
                            "continuing_particle" in evidence.reasons
                            and "new_sentence_connector" not in evidence.reasons
                        )
                        if hard_continuation:
                            cost += 0.72
                        elif "terminal_particle_predicate_follows" in evidence.reasons:
                            cost += 0.24
                        elif "continuative_inflection" in evidence.reasons:
                            cost += 0.34
                        if evidence.sentence_type == "continuation" and acoustic < 0.7:
                            cost += 0.16
                        elif evidence.sentence_type == "fragment" and acoustic < 0.85:
                            cost += 0.12

                    segment = JointSegment(
                        token_start,
                        token_end,
                        atom_start,
                        atom_end,
                        round(ratio, 6),
                        round(boundary_score, 6),
                    )
                    candidate = (
                        cost,
                        prior_count + 1,
                        (token_start, atom_start),
                        segment,
                    )
                    if winner is None or (
                        round(candidate[0], 12), candidate[1]
                    ) < (round(winner[0], 12), winner[1]):
                        winner = candidate
                if winner is not None:
                    best[(token_end, atom_end)] = winner

        final_key = (token_count, atom_count)
        if final_key not in best:
            return None
        score, count, _, _ = best[final_key]
        segments = []
        key = final_key
        while key != (0, 0):
            _, _, predecessor, segment = best[key]
            if predecessor is None or segment is None:
                raise DetectError(
                    "DETECT_ALIGNMENT_INVALID",
                    "Joint partition lost its predecessor",
                )
            segments.append(segment)
            key = predecessor
        segments.reverse()
        if len(segments) != count:
            raise DetectError(
                "DETECT_ALIGNMENT_INVALID",
                "Joint partition count is inconsistent",
            )
        return JointPartition(tuple(segments), round(score, 6))


REGION_PARTITIONERS = {
    "joint-dynamic-programming": JointDynamicProgrammingPartitioner,
}


def create_region_partitioner(name: str, config: DetectConfig) -> RegionPartitioner:
    try:
        return REGION_PARTITIONERS[name](config)
    except KeyError as exc:
        raise DetectError(
            "DETECT_IMPLEMENTATION_UNKNOWN",
            f"Unknown region partitioner: {name}",
        ) from exc
