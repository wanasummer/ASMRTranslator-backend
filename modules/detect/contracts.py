"""Stable contracts for Japanese sentence detection between ASR and cleanup."""

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..vad.contracts import AsrSlice, VadRegion


class DetectError(RuntimeError):
    """Domain error raised at the detect-stage boundary."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class AsrBlock(Protocol):
    """Minimum ASR-stage contract consumed by detect."""

    index: int
    start_ms: int
    end_ms: int
    text: str
    skip_tts: bool
    asr_slice: AsrSlice | None
    source_text: str
    cleaned_source_text: str
    translation_available: bool
    aligned_sentence: object | None


@dataclass(frozen=True)
class JapaneseToken:
    surface: str
    start_char: int
    end_char: int
    pos: str = ""
    pos_detail: str = ""
    conjugation_type: str = ""
    conjugation_form: str = ""
    reading: str = ""
    mora: float = 0.0
    unknown: bool = False


@dataclass(frozen=True)
class SentenceBoundary:
    """Language evidence after ``token_index``; detectors emit every token end."""

    token_index: int
    char_index: int
    language_score: float
    sentence_type: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SentenceDetectionInput:
    parent: AsrSlice
    source_text: str


@dataclass(frozen=True)
class SentenceDetection:
    normalized_text: str
    tokens: tuple[JapaneseToken, ...]
    boundaries: tuple[SentenceBoundary, ...]
    method: str
    parse_confidence: float
    reason: str | None = None


class SentenceDetector(Protocol):
    def detect(self, value: SentenceDetectionInput) -> SentenceDetection: ...


@dataclass(frozen=True)
class AlignmentRequest:
    parent: AsrSlice
    source_block_id: str
    detection: SentenceDetection
    bypass_reason: str | None = None


@dataclass(frozen=True)
class AlignedSentenceBlock:
    id: str
    source_block_id: str
    start_ms: int
    end_ms: int
    text: str
    source_region_ids: tuple[str, ...]
    method: str
    confidence: str
    sentence_type: str = "unknown"


@dataclass(frozen=True)
class AlignmentResult:
    blocks: tuple[AlignedSentenceBlock, ...]
    attempts: tuple[dict[str, object], ...]
    outcome: str
    reason: str | None = None
    timings: tuple[dict[str, object], ...] = field(default=(), compare=False)


class SentenceAligner(Protocol):
    """Aligners return a stable outcome: exact/aligned/fallback/bypassed."""

    def align(self, value: AlignmentRequest) -> AlignmentResult: ...


class SpeechAtomDetector(Protocol):
    def detect(self, parent: AsrSlice, threshold_ms: int) -> tuple[VadRegion, ...]: ...


class SentenceDurationEstimator(Protocol):
    def estimate_units(self, tokens: Sequence[JapaneseToken]) -> float: ...


@dataclass(frozen=True)
class JointSegment:
    token_start: int
    token_end: int
    atom_start: int
    atom_end: int
    duration_ratio: float
    boundary_score: float


@dataclass(frozen=True)
class JointPartition:
    segments: tuple[JointSegment, ...]
    score: float


class RegionPartitioner(Protocol):
    def partition(
        self,
        detection: SentenceDetection,
        regions: Sequence[VadRegion],
        estimator: SentenceDurationEstimator,
    ) -> JointPartition | None: ...


@dataclass(frozen=True)
class AlignmentDependencies:
    """Adapters available to any aligner, including future forced aligners."""

    audio_path: str
    atom_detector: SpeechAtomDetector
    duration_estimator: SentenceDurationEstimator
    region_partitioner: RegionPartitioner


@dataclass(frozen=True)
class DetectConfig:
    sentence_detector: str = "mecab-japanese"
    aligner: str = "joint-vad"
    qwen_endpoint: str = "http://127.0.0.1:8007"
    vad_endpoint: str = "http://127.0.0.1:8006"
    vad_threshold: float = 0.5
    atom_detector: str = "native-vad"
    duration_estimator: str = "japanese-mora"
    region_partitioner: str = "joint-dynamic-programming"
    min_region_duration_ms: int = 100
    atom_silence_ms: int = 200
    min_sentence_duration_ms: int = 280
    max_sentence_duration_ms: int = 12000
    max_sentence_count: int = 12
    min_sentence_units: float = 1.5
    max_duration_ratio_error: float = 3.0
    boundary_bias: float = 0.72
    language_weight: float = 0.55
    acoustic_weight: float = 0.75
    boundary_synergy_weight: float = 0.30
    duration_weight: float = 0.65
    unknown_language_floor: float = 0.12
    vocalization_chars: str = "嗯啊阿哈哦喔呃唔呀哼欸诶あぁうぅんはぁえぇおぉっッー"
    vocalization_ratio: float = 0.72
    vocalization_run: int = 5

    def __post_init__(self):
        positive = (
            self.min_region_duration_ms,
            self.atom_silence_ms,
            self.min_sentence_duration_ms,
            self.max_sentence_duration_ms,
            self.max_sentence_count,
            self.min_sentence_units,
            self.max_duration_ratio_error,
        )
        weights = (
            self.boundary_bias,
            self.language_weight,
            self.acoustic_weight,
            self.boundary_synergy_weight,
            self.duration_weight,
        )
        if (
            any(value <= 0 for value in positive)
            or self.max_sentence_duration_ms < self.min_sentence_duration_ms
            or self.max_duration_ratio_error <= 1
            or any(value < 0 for value in weights)
            or not 0 <= self.unknown_language_floor <= 1
            or not 0 < self.vocalization_ratio <= 1
            or self.vocalization_run <= 0
        ):
            raise ValueError("Invalid detect-stage configuration")


@dataclass(frozen=True)
class DetectStageInput:
    """ASR blocks, copied on output; absent slice lineage uses authored ranges.

    Document ASR speaker turns may overlap and retain their source ordering.
    """

    asr_blocks: tuple[AsrBlock, ...]


@dataclass(frozen=True)
class DetectStageResult:
    blocks: tuple[AsrBlock, ...]
    summary: dict[str, int]


class DetectStage(Protocol):
    def run(self, value: DetectStageInput, output_dir: str) -> DetectStageResult: ...
