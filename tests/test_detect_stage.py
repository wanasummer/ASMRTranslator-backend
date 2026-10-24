"""Japanese sentence detection, acoustic alignment, and pipeline handoff."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from modules.detect.aligners import JointVadSentenceAligner
from modules.detect.contracts import (
    AlignedSentenceBlock,
    AlignmentDependencies,
    AlignmentRequest,
    DetectConfig,
    DetectError,
    DetectStageInput,
    DetectStageResult,
    SentenceDetectionInput,
)
from modules.detect.duration import (
    JapaneseMoraDurationEstimator,
    count_japanese_mora,
)
from modules.detect.factory import create_detect_stage
from modules.detect.partitioners import JointDynamicProgrammingPartitioner
from modules.detect.sentence_detectors import (
    ConservativeJapaneseTokenizer,
    MecabJapaneseSentenceDetector,
)
from modules.detect.stage import DefaultDetectStage, detect_eligible
from modules.vad.analysis import analyze_regions
from modules.vad.contracts import AsrSlice, VadRegion
from modules.vad.models import SpeechRegion
from modules.vtt_parser import VTTBlock


def region(name, start, end):
    return VadRegion(str(name), start, end)


def parent_slice(start=0, end=10000):
    return AsrSlice("p", start, end, start, end, ("r",))


def analyze(text, parent=None, *, conservative=False):
    tokenizer = ConservativeJapaneseTokenizer() if conservative else None
    detector = MecabJapaneseSentenceDetector(tokenizer)
    return detector.detect(SentenceDetectionInput(parent or parent_slice(), text))


def align(text, atoms, *, conservative=False, config=None):
    config = config or DetectConfig()
    parent = parent_slice(atoms[0].speech_start_ms, atoms[-1].speech_end_ms)
    detection = analyze(text, parent, conservative=conservative)
    atom_detector = Mock()
    atom_detector.detect.return_value = atoms
    aligner = JointVadSentenceAligner(config, AlignmentDependencies(
        "unused.wav",
        atom_detector,
        JapaneseMoraDurationEstimator(),
        JointDynamicProgrammingPartitioner(config),
    ))
    return aligner.align(AlignmentRequest(parent, parent.id, detection)), atom_detector


class JapaneseSentenceDetectorTests(unittest.TestCase):
    def test_mora_uses_reading_and_unknown_kanji_fallback(self):
        self.assertEqual(count_japanese_mora("キョー"), 2)
        self.assertEqual(count_japanese_mora("ガッコー"), 4)
        self.assertEqual(count_japanese_mora("", "未知"), 4)

    def test_terminal_particle_absorbs_preceding_auxiliary(self):
        detection = analyze("今日はいい天気ですね明日はどこへ行きますか")
        surfaces = [token.surface for token in detection.tokens]
        after_desu = detection.boundaries[surfaces.index("です")]
        after_ne = detection.boundaries[surfaces.index("ね")]
        after_masu = detection.boundaries[surfaces.index("ます")]
        self.assertLess(after_desu.language_score, 0.1)
        self.assertGreater(after_ne.language_score, 0.8)
        self.assertLess(after_masu.language_score, 0.1)
        self.assertEqual(detection.boundaries[-1].sentence_type, "question")

    def test_connector_is_found_even_when_mecab_splits_demo(self):
        detection = analyze("朝早いんだからでも今日は休み")
        surfaces = [token.surface for token in detection.tokens]
        after_kara = max(
            boundary for boundary in detection.boundaries
            if surfaces[boundary.token_index - 1] == "から"
        )
        self.assertGreaterEqual(after_kara.language_score, 0.75)
        self.assertIn("new_sentence_connector", after_kara.reasons)

    def test_auxiliary_particle_inside_nandaka_is_not_a_question_boundary(self):
        detection = analyze("目を覚ますなんだか今日はよく寝たみたい")
        surfaces = [token.surface for token in detection.tokens]
        after_kaku = detection.boundaries[surfaces.index("覚ます")]
        after_da = detection.boundaries[surfaces.index("だ")]
        after_ka = detection.boundaries[surfaces.index("か")]
        self.assertGreaterEqual(after_kaku.language_score, 0.5)
        self.assertIn("new_clause_starter", after_kaku.reasons)
        self.assertLess(after_da.language_score, 0.1)
        self.assertIn("fixed_expression_continuation", after_da.reasons)
        self.assertLess(after_ka.language_score, 0.2)
        self.assertNotIn("terminal_particle", after_ka.reasons)

    def test_asr_word_fragment_does_not_override_nominal_no_ending(self):
        detection = analyze("もう朝なの全身のけだらさを覚えて目を覚ます")
        surfaces = [token.surface for token in detection.tokens]
        after_no = detection.boundaries[surfaces.index("の")]
        after_noke = detection.boundaries[surfaces.index("のけ")]
        self.assertGreaterEqual(after_no.language_score, 0.6)
        self.assertIn("nominal_no_ending", after_no.reasons)
        self.assertLess(after_noke.language_score, 0.1)
        self.assertIn("suspect_asr_word_fragment", after_noke.reasons)

    def test_fixed_expressions_are_not_split_inside(self):
        nandakedo = analyze("そのことなんだけどどうかしたの")
        surfaces = [token.surface for token in nandakedo.tokens]
        after_da = nandakedo.boundaries[surfaces.index("だ")]
        self.assertLess(after_da.language_score, 0.1)
        self.assertIn("fixed_expression_continuation", after_da.reasons)

        nantoka = analyze("なんとか平静を装っていた")
        surfaces = [token.surface for token in nantoka.tokens]
        after_ka = nantoka.boundaries[surfaces.index("か")]
        self.assertLess(after_ka.language_score, 0.2)
        self.assertNotIn("quotative_terminal", after_ka.reasons)

    def test_informal_toka_keeps_weak_boundary_evidence(self):
        detection = analyze("嬉しいしとか用事がある")
        surfaces = [token.surface for token in detection.tokens]
        after_ka = detection.boundaries[surfaces.index("か")]
        self.assertGreaterEqual(after_ka.language_score, 0.45)
        self.assertIn("quotative_terminal", after_ka.reasons)

    def test_unknown_asr_has_low_parse_confidence_but_keeps_cut_positions(self):
        detection = analyze("謎語XYZ謎語", conservative=True)
        self.assertLess(detection.parse_confidence, 0.2)
        self.assertEqual(len(detection.tokens), len(detection.normalized_text))
        self.assertEqual(detection.boundaries[-1].char_index, len(detection.normalized_text))

    def test_tokenizer_runtime_failure_degrades_per_slice(self):
        tokenizer = Mock()
        tokenizer.name = "broken"
        tokenizer.tokenize.side_effect = RuntimeError("bad ASR text")
        detection = MecabJapaneseSentenceDetector(tokenizer).detect(
            SentenceDetectionInput(parent_slice(), "聞き取れないXYZ")
        )
        self.assertIn("runtime-fallback", detection.method)
        self.assertTrue(detection.tokens)
        self.assertLess(detection.parse_confidence, 0.2)


class JointVadAlignerTests(unittest.TestCase):
    def test_slow_nominal_question_does_not_pull_text_across_pause(self):
        atoms = tuple(region(f"a{i}", start, end) for i, (start, end) in enumerate([
            (8770, 9620),
            (9620, 10970),
            (11730, 14630),
            (14800, 15520),
            (15520, 16510),
            (16510, 17580),
        ]))
        text = (
            "もう朝なの全身のけだらさを覚えて目を覚ます"
            "なんだか今日はたっぷり寝てたみたい"
        )
        result, _ = align(text, atoms)
        self.assertEqual([block.text for block in result.blocks], [
            "もう朝なの",
            "全身のけだらさを覚えて目を覚ます",
            "なんだか今日はたっぷり寝てたみたい",
        ])
        self.assertEqual(
            [(block.start_ms, block.end_ms) for block in result.blocks],
            [(8770, 10970), (11730, 14630), (14800, 17580)],
        )

    def test_reported_case_is_split_by_japanese_semantics_and_audio(self):
        atoms = tuple(region(f"a{i}", start, end) for i, (start, end) in enumerate([
            (74970, 76390),
            (76710, 79190),
            (80500, 81630),
            (82610, 84910),
            (85270, 86770),
            (87390, 90060),
        ]))
        # Keep the actual punctuation-free ASR text, including its recognition
        # errors.  The stray ``よ`` must not create a short false sentence.
        text = (
            "どこがよ変わらず朝はんだから"
            "でも今日から本格的な夏休みなんかドキドキする"
            "どこに遊びに行こうかしらミヤ"
        )
        result, detector = align(text, atoms)
        self.assertEqual([block.text for block in result.blocks], [
            "どこがよ変わらず朝はんだから",
            "でも今日から本格的な夏休みなんかドキドキする",
            "どこに遊びに行こうかしらミヤ",
        ])
        self.assertEqual(
            [(block.start_ms, block.end_ms) for block in result.blocks],
            [(74970, 79190), (80500, 86770), (87390, 90060)],
        )
        detector.detect.assert_called_once_with(parent_slice(74970, 90060), 200)

    def test_weak_language_and_200ms_gap_remain_one_sentence(self):
        atoms = (region("a", 0, 1000), region("b", 1200, 2200))
        result, _ = align("abcdefghij", atoms, conservative=True)
        self.assertEqual(len(result.blocks), 1)

    def test_long_gap_recovers_boundary_when_asr_has_no_punctuation(self):
        atoms = (region("a", 0, 1000), region("b", 2300, 3300))
        result, _ = align("abcdefghij", atoms, conservative=True)
        self.assertEqual(len(result.blocks), 2)
        self.assertEqual("".join(block.text for block in result.blocks), "abcdefghij")

    def test_explicit_japanese_punctuation_can_use_short_gap(self):
        atoms = (region("a", 0, 1100), region("b", 1300, 2500))
        result, _ = align("今日はいい。明日も行く。", atoms)
        self.assertEqual([block.text for block in result.blocks], [
            "今日はいい。", "明日も行く。",
        ])

    def test_every_atom_and_token_is_owned_once_without_mutation(self):
        atoms = (
            region("a", 0, 1000),
            region("b", 1800, 2800),
            region("c", 3600, 5000),
        )
        parent = parent_slice(0, 5000)
        detection = analyze("今日は休みですね明日は遊びますか", parent)
        request = AlignmentRequest(parent, parent.id, detection)
        before = copy.deepcopy(request)
        detector = Mock()
        detector.detect.return_value = atoms
        config = DetectConfig()
        result = JointVadSentenceAligner(config, AlignmentDependencies(
            "unused.wav",
            detector,
            JapaneseMoraDurationEstimator(),
            JointDynamicProgrammingPartitioner(config),
        )).align(request)
        self.assertEqual(request, before)
        self.assertEqual("".join(block.text for block in result.blocks), detection.normalized_text)
        self.assertEqual(
            [item for block in result.blocks for item in block.source_region_ids],
            [item.id for item in atoms],
        )

    def test_atom_failure_degrades_to_real_unpadded_parent(self):
        parent = AsrSlice("p", 0, 5000, 400, 4600, ("r",))
        detection = analyze("今日は休みです", parent)
        detector = Mock()
        detector.detect.side_effect = RuntimeError("native failure")
        config = DetectConfig()
        aligner = JointVadSentenceAligner(config, AlignmentDependencies(
            "unused.wav",
            detector,
            JapaneseMoraDurationEstimator(),
            JointDynamicProgrammingPartitioner(config),
        ))
        result = aligner.align(AlignmentRequest(parent, parent.id, detection))
        self.assertEqual(result.reason, "atom_detection_failed")
        self.assertEqual((result.blocks[0].start_ms, result.blocks[0].end_ms), (400, 4600))

    def test_config_rejects_invalid_values(self):
        for values in (
            {"atom_silence_ms": 0},
            {"max_duration_ratio_error": 1},
            {"boundary_bias": -1},
            {"vocalization_ratio": 0},
        ):
            with self.assertRaises(ValueError):
                DetectConfig(**values)


class DetectStageTests(unittest.TestCase):
    def test_stage_returns_japanese_sentences_for_cleaner_and_writes_audit(self):
        parent = parent_slice(0, 3300)
        source = VTTBlock(
            0,
            0,
            3300,
            "今日は休みです。明日は仕事です。",
            asr_slice=parent,
            source_text="今日は休みです。明日は仕事です。",
        )
        atoms = (region("a", 0, 1400), region("b", 1900, 3300))
        atom_detector = Mock()
        atom_detector.detect.return_value = atoms
        config = DetectConfig()
        stage = DefaultDetectStage(
            config,
            MecabJapaneseSentenceDetector(),
            JointVadSentenceAligner(config, AlignmentDependencies(
                "unused.wav",
                atom_detector,
                JapaneseMoraDurationEstimator(),
                JointDynamicProgrammingPartitioner(config),
            )),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = stage.run(DetectStageInput((source,)), directory)
            audit = json.loads(
                (Path(directory) / "detect_output" / "sentence_alignment.json")
                .read_text(encoding="utf-8")
            )
        self.assertEqual(source.text, "今日は休みです。明日は仕事です。")
        self.assertEqual([block.text for block in result.blocks], [
            "今日は休みです。", "明日は仕事です。",
        ])
        self.assertTrue(all(not block.cleaned_source_text for block in result.blocks))
        self.assertTrue(all(not block.translation_available for block in result.blocks))
        self.assertEqual(audit["summary"]["outputSentences"], 2)
        self.assertEqual(audit["slices"][0]["detection"]["method"], "mecab-unidic-lite")

    def test_factory_selection_is_replaceable(self):
        stage = create_detect_stage(DetectConfig(), "source.wav")
        self.assertIsInstance(stage, DefaultDetectStage)
        self.assertEqual(stage.aligner.audio_path, "source.wav")
        with self.assertRaisesRegex(DetectError, "DETECT_IMPLEMENTATION_UNKNOWN"):
            create_detect_stage(DetectConfig(aligner="qwen-force-align"), "unused.wav")

    def test_eligibility_is_beta_gated_but_does_not_depend_on_llm(self):
        args = dict(
            enabled=True,
            subtitle_path="",
            backend="fish-asr",
            language="ja",
            project_vad=True,
        )
        self.assertTrue(detect_eligible(**args))
        self.assertFalse(detect_eligible(**{**args, "enabled": False}))
        self.assertFalse(detect_eligible(**{**args, "subtitle_path": "input.vtt"}))


class PipelineHandoffTests(unittest.TestCase):
    def test_detect_runs_before_cleaner_and_translation_keeps_its_timeline(self):
        from modules.pipeline import run_pipeline
        from modules.japanese_cleaning.mecab_regex import RegexJapaneseCleaner

        slices, analysis = analyze_regions([SpeechRegion(1, 3)], {})
        parent = VTTBlock(
            0,
            600,
            3400,
            "これは日本語です。",
            asr_slice=analysis.slices[0],
            source_text="これは日本語です。",
        )
        sentence_slice = AsrSlice("detected", 1111, 2222, 1111, 2222, ("atom",))
        detected_block = replace(
            parent,
            start_ms=1111,
            end_ms=2222,
            text="検出した日本語。",
            source_text="検出した日本語。",
            asr_slice=sentence_slice,
            aligned_sentence=AlignedSentenceBlock(
                "detected",
                analysis.slices[0].id,
                1111,
                2222,
                "検出した日本語。",
                ("atom",),
                "fake",
                "high",
                "statement",
            ),
        )
        fake_stage = Mock()
        fake_stage.run.return_value = DetectStageResult(
            (detected_block,), {"alignedSlices": 1}
        )
        cleaner = RegexJapaneseCleaner()
        cleaner_inputs, translator_inputs, captured = [], [], []

        class RecordingCleaner:
            name = cleaner.name

            def clean(self, blocks):
                cleaner_inputs.extend(blocks)
                return cleaner.clean(blocks)

        class StopAfterHandoff(Exception):
            pass

        def translate(blocks, config, **kwargs):
            translator_inputs.extend(blocks)
            self.assertTrue(kwargs["preserve_source_timeline"])
            return [replace(
                blocks[0], text="检测后的翻译。", translation_available=True
            )], {"enabled": True}

        def split(audio_path, blocks, *args, **kwargs):
            captured.extend(blocks)
            raise StopAfterHandoff()

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch(
                "modules.vad.stage.run_analysis_stage",
                return_value=(slices, analysis),
            ))
            stack.enter_context(patch("modules.asr.create_asr", return_value=object()))
            stack.enter_context(patch(
                "modules.asr_engine.transcribe_slices", return_value=[parent]
            ))
            stack.enter_context(patch(
                "modules.pipeline._create_japanese_cleaner_with_fallback",
                return_value=RecordingCleaner(),
            ))
            stack.enter_context(patch(
                "modules.llm_enhancer.enhance_asr_blocks", side_effect=translate
            ))
            stack.enter_context(patch(
                "modules.detect.create_detect_stage", return_value=fake_stage
            ))
            stack.enter_context(patch("modules.pipeline.get_audio_duration", return_value=8))
            stack.enter_context(patch("modules.pipeline.split_audio", side_effect=split))
            from modules.error import PipelineError
            with self.assertRaises(PipelineError):
                run_pipeline(
                    "test.wav",
                    "",
                    directory,
                    asr_kwargs={"api_key": "test"},
                    llm_kwargs={"api_key": "test", "model": "test"},
                    detect_enabled=True,
                )

        detect_input = fake_stage.run.call_args.args[0].asr_blocks
        self.assertEqual([block.text for block in detect_input], ["これは日本語です。"])
        self.assertEqual([block.text for block in cleaner_inputs], ["検出した日本語。"])
        self.assertEqual([block.text for block in translator_inputs], ["検出した日本語。"])
        self.assertEqual(
            [(block.start_ms, block.end_ms, block.text) for block in captured],
            [(1111, 2222, "检测后的翻译。")],
        )


if __name__ == "__main__":
    unittest.main()
