import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from modules.arrangement import (ArrangementError, initial_state, load_state, save_state,
                                 optimize_subtitles, synthesize_edit, prepare_arrangement_plan,
                                 resolve_replacement, run_editor_request, optimize_fragment,
                                 pipeline_tts_rates)
from modules.arrangement_audio import ArrangementPlan, ArrangementAudioResult, PydubArrangementAdapter, perceived_db, write_arrangement_result
from modules.arrangement_contracts import ArrangedClip
from modules.llm_enhancer import LLMClient, LLMConfig, build_phase2_system, _parse_single_optimized_text
from modules.pipeline import run_arrangement_pipeline


class ArrangementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        (self.work / "debug").mkdir()
        self.rows = [{"index": 7, "start_ms": 0, "end_ms": 1000, "text": "请听我说一句话", "source_text": "聞いてください"},
                     {"index": 13, "start_ms": 2000, "end_ms": 3000, "text": "我们慢慢说话吧", "source_text": "ゆっくり話しましょう"}]
        (self.work / "debug/_parsed_vtt.json").write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")
        self.state = initial_state(self.work)

    def test_default_interval_uses_completed_pipeline_tempo(self):
        (self.work / "pipeline_20260101_010101.log").write_text(
            "[budget] seg 13: 人声 2000ms > 预算 1000ms → 加速1.25x 至 1000ms\n"
            "Pipeline complete!\n", encoding="utf-8")
        self.assertEqual(pipeline_tts_rates(self.work), {13: 1.25})
        cache = self.work / "chunk_0/02_tts"
        cache.mkdir(parents=True)
        (cache / "tts_tempo_0013.json").write_text('{"rate": 1.2}', encoding="utf-8")
        self.assertEqual(pipeline_tts_rates(self.work), {13: 1.2})
        (cache / "tts_0000.wav").touch()
        (cache / "tts_0001.wav").touch()
        with patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 2000)):
            plan = prepare_arrangement_plan(self.work, Path("original.wav"), self.state, {}, {})
        self.assertEqual([(clip.start_ms, clip.end_ms) for clip in plan.clips], [(0, 2000), (2000, 3667)])

    def test_tts_sample_changes_plan_source_without_changing_saved_tempo(self):
        cache = self.work / "chunk_0/02_tts"
        cache.mkdir(parents=True)
        (cache / "tts_0000.wav").touch()
        (cache / "tts_0001.wav").touch()
        self.state["segments"][0]["ttsSample"] = {"start": 250, "end": 1250}
        with patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 2000)):
            plan = prepare_arrangement_plan(self.work, Path("original.wav"), self.state, {}, {})
        self.assertEqual((plan.clips[0].sample_start_ms, plan.clips[0].sample_end_ms), (250, 1250))
        self.assertEqual((plan.clips[0].start_ms, plan.clips[0].end_ms), (0, 1000))
        self.state["segments"][0]["ttsSample"] = {"start": 250, "end": 2250}
        with patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 2000)):
            with self.assertRaises(ArrangementError):
                prepare_arrangement_plan(self.work, Path("original.wav"), self.state, {}, {})

    def test_global_loudness_is_a_multiplier_of_each_fragment_adjustment(self):
        cache = self.work / "chunk_0/02_tts"
        cache.mkdir(parents=True)
        (cache / "tts_0000.wav").touch()
        (cache / "tts_0001.wav").touch()
        self.state["segments"][0]["volume"] = .5
        self.state["segments"][0]["originalInterval"] = {"start": 100, "end": 900}
        self.state["segments"][1]["volume"] = 1.5
        self.state.update(loudnessScope="global", globalVolume=1.3)
        with patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 1000)):
            plan = prepare_arrangement_plan(self.work, Path("original.wav"), self.state, {}, {})
        self.assertEqual([round(clip.volume, 2) for clip in plan.clips], [.65, 1.95])
        self.assertEqual((plan.clips[0].reference_start_ms, plan.clips[0].reference_end_ms), (100, 900))
        self.assertFalse(plan.global_loudness)
        self.state["loudnessScope"] = "segment"
        with patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 1000)):
            segment_scope_plan = prepare_arrangement_plan(self.work, Path("original.wav"), self.state, {}, {})
        self.assertEqual([round(clip.volume, 2) for clip in segment_scope_plan.clips], [.65, 1.95])

    def test_save_restores_all_edits_without_synthesizing(self):
        self.state["segments"][0].update(text="新草稿", japanese="新しい原文", volume=.5,
                                          interval={"start": 100, "end": 900})
        self.state.update(loudnessScope="global", globalVolume=1.3)
        with patch("modules.arrangement.synthesize_edit") as synth:
            run_editor_request({"workDir": str(self.work), "action": "save", "state": self.state})
        synth.assert_not_called()
        self.assertEqual(load_state(self.work), self.state)
        self.assertEqual(json.loads((self.work / "debug/_parsed_vtt.json").read_text(encoding="utf-8")), self.rows)

    def test_deleted_fragment_stays_in_draft_but_is_omitted_from_audio_plan(self):
        cache = self.work / "chunk_0/02_tts"
        cache.mkdir(parents=True)
        (cache / "tts_0001.wav").touch()
        self.state["segments"][0]["deleted"] = True
        self.state["segments"][1]["interval"] = {"start": -7000, "end": -5000}
        self.state["segments"][1]["originalInterval"] = {"start": 1800, "end": 3200}
        save_state(self.work, self.state)
        self.assertTrue(load_state(self.work)["segments"][0]["deleted"])
        with patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 1000)):
            plan = prepare_arrangement_plan(self.work, Path("original.wav"), load_state(self.work), {}, {})
        self.assertEqual([clip.index for clip in plan.clips], [13])
        self.assertEqual((plan.clips[0].start_ms, plan.clips[0].end_ms), (-7000, -5000))
        self.assertEqual((plan.clips[0].reference_start_ms, plan.clips[0].reference_end_ms), (1800, 3200))
        self.state["segments"][0]["deleted"] = False
        self.state["segments"][1]["interval"] = None
        save_state(self.work, self.state)
        self.assertFalse(load_state(self.work)["segments"][0]["deleted"])

    def test_stale_source_and_invalid_intervals_are_rejected(self):
        bad = json.loads(json.dumps(self.state))
        bad["segments"][0]["interval"] = {"start": 0, "end": 2200}
        bad["segments"][1]["interval"] = {"start": 2000, "end": 3000}
        with self.assertRaises(ArrangementError):
            save_state(self.work, bad)
        bad["segments"][0]["interval"] = {"start": -8000, "end": -5001}
        bad["segments"][1]["interval"] = None
        with self.assertRaises(ArrangementError):
            save_state(self.work, bad)
        (self.work / "debug/_parsed_vtt.json").write_text("[]")
        with self.assertRaises(ArrangementError):
            save_state(self.work, self.state)

    def test_load_rebases_stale_draft_with_same_segment_ids_and_preserves_backup(self):
        self.state["segments"][0].update(text="用户修改的字幕", volume=.7,
                                          interval={"start": 100, "end": 900})
        save_state(self.work, self.state)
        old_version = self.state["sourceVersion"]
        self.rows[0]["text"] = "重新生成的字幕"
        (self.work / "debug/_parsed_vtt.json").write_text(
            json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")

        result = run_editor_request({"workDir": str(self.work), "action": "load"})["state"]
        self.assertNotEqual(result["sourceVersion"], old_version)
        self.assertEqual(result["segments"][0]["text"], "用户修改的字幕")
        self.assertEqual(result["segments"][0]["volume"], .7)
        self.assertEqual(result["segments"][0]["interval"], {"start": 100, "end": 900})
        draft_path = self.work / "arrangement/draft.json"
        self.assertEqual(json.loads(draft_path.read_text(encoding="utf-8"))["sourceVersion"], old_version)

        save_state(self.work, result)
        backups = list((self.work / "arrangement").glob("draft.stale.*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8"))["sourceVersion"], old_version)
        self.assertEqual(load_state(self.work), result)

    def test_changed_segment_ids_start_fresh_without_overwriting_old_draft(self):
        self.state["segments"][0]["text"] = "保留在旧草稿中的字幕"
        save_state(self.work, self.state)
        self.rows[0]["index"] = 99
        (self.work / "debug/_parsed_vtt.json").write_text(
            json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")

        result = load_state(self.work)
        self.assertEqual(result["segments"][0]["index"], 99)
        self.assertEqual(result["segments"][0]["text"], self.rows[0]["text"])
        self.assertEqual(json.loads((self.work / "arrangement/draft.json").read_text(encoding="utf-8"))["segments"][0]["text"], "保留在旧草稿中的字幕")

    def test_single_optimization_always_uses_current_japanese_and_does_not_mutate(self):
        optimizer = MagicMock()
        optimizer.optimize_from_japanese = AsyncMock(side_effect=["新译文一", "新译文二"])
        first = asyncio.run(optimize_subtitles(self.state, optimizer, 0))
        second = asyncio.run(optimize_subtitles(first, optimizer, 0))
        self.assertEqual(second["segments"][0]["text"], "新译文二")
        self.assertEqual(self.state["segments"][0]["text"], self.rows[0]["text"])
        self.assertEqual([call.args[1] for call in optimizer.optimize_from_japanese.call_args_list], ["聞いてください"] * 2)
        self.assertEqual(second["segments"][1], self.state["segments"][1])

    def test_global_optimization_translates_japanese_or_polishes_chinese_and_preserves_ids(self):
        optimizer = MagicMock()
        optimizer.optimize_from_japanese = AsyncMock(return_value="从日语生成中文")
        optimizer.optimize_chinese_text = AsyncMock(return_value="另一段中文")
        self.state["segments"][1]["japanese"] = ""
        result = asyncio.run(optimize_subtitles(self.state, optimizer))
        self.assertEqual(optimizer.optimize_chinese_text.call_args.args[1], self.rows[1]["text"])
        optimizer.optimize_from_japanese.assert_awaited_once()
        self.assertEqual(result["segments"][0]["text"], "从日语生成中文")
        self.assertEqual([row["index"] for row in result["segments"]], [7, 13])

    def test_text_optimization_keeps_languages_separate(self):
        optimizer = MagicMock()
        optimizer.optimize_chinese_text = AsyncMock(return_value="优化的中文")
        optimizer.optimize_japanese_text = AsyncMock(return_value="聞いてください。")
        original = self.state["segments"][0]
        chinese = asyncio.run(optimize_fragment(original, optimizer, "chinese"))
        japanese = asyncio.run(optimize_fragment(original, optimizer, "japanese"))
        self.assertEqual(chinese["japanese"], original["japanese"])
        self.assertEqual(japanese["text"], original["text"])
        self.assertEqual(japanese["japanese"], "聞いてください。")
        self.assertEqual(original["text"], self.rows[0]["text"])

    def test_fragment_endpoint_returns_only_its_result_without_overwriting_project_draft(self):
        save_state(self.work, self.state)
        client = MagicMock()
        client.optimize_from_japanese = AsyncMock(return_value="新的中文")
        payload = {"workDir": str(self.work), "action": "optimize-fragment", "position": 0,
                   "sourceVersion": self.state["sourceVersion"], "fragment": self.state["segments"][0],
                   "mode": "auto", "config": {}}
        with patch("modules.arrangement.create_optimizer", return_value=client):
            result = run_editor_request(payload)
        self.assertEqual(result["edit"]["text"], "新的中文")
        self.assertEqual(load_state(self.work), self.state)
        self.assertEqual(len(list((self.work / "arrangement/audit").glob("*.json"))), 2)

    def test_japanese_text_service_never_requests_chinese_translation(self):
        from modules.vtt_parser import VTTBlock
        client = LLMClient(LLMConfig())
        client._post = AsyncMock(return_value="7\t聞いてください。")
        self.assertEqual(asyncio.run(client.optimize_japanese_text(VTTBlock(7, 0, 1, "中文"), "聞いてください")), "聞いてください。")
        self.assertIn("不翻译成中文", client._post.call_args.args[0])

    def test_single_optimization_accepts_plain_text_or_model_renumbering(self):
        from modules.vtt_parser import VTTBlock
        for response in ("啊，烦死人了。", "1\t啊，烦死人了。", "```text\n啊，烦死人了。\n```", '{"text":"啊，烦死人了。"}'):
            client = LLMClient(LLMConfig())
            client._post = AsyncMock(return_value=response)
            self.assertEqual(asyncio.run(client.optimize_chinese_text(VTTBlock(0, 0, 1, "啊...烦死人了"), "啊...烦死人了")), "啊，烦死人了。")
            self.assertNotIn("0\t", client._post.call_args.args[1])
        self.assertEqual(_parse_single_optimized_text("2026 年见"), "2026 年见")
        self.assertEqual(_parse_single_optimized_text("3.14米"), "3.14米")
        self.assertEqual(_parse_single_optimized_text("1\t第一段\n2\t第二段"), "")

    def test_background_commit_preserves_other_edits_and_rejects_changed_text(self):
        save_state(self.work, self.state)
        expected = json.loads(json.dumps(self.state["segments"][0]))
        optimized = {**expected, "text": "后台优化后的文本"}
        self.state["segments"][1]["volume"] = .4
        save_state(self.work, self.state)
        payload = {"workDir": str(self.work), "action": "commit-optimization", "position": 0,
                   "sourceVersion": self.state["sourceVersion"], "fragment": expected,
                   "optimizedFragment": optimized, "mode": "chinese"}
        result = run_editor_request(payload)
        self.assertTrue(result["applied"])
        self.assertEqual(load_state(self.work)["segments"][1]["volume"], .4)
        latest = load_state(self.work)
        latest["segments"][0]["text"] = "用户随后编辑的文本"
        save_state(self.work, latest)
        self.assertFalse(run_editor_request(payload)["applied"])
        self.assertEqual(load_state(self.work)["segments"][0]["text"], "用户随后编辑的文本")

    def test_regeneration_creates_fresh_audio_assets_and_keeps_text_provenance(self):
        adapter = MagicMock()
        adapter.name = "edge-tts"
        adapter.output_extension = ".wav"
        adapter.get_voice_for_speaker.return_value = "voice"
        async def synth(text, voice, path, **kwargs):
            Path(path).write_bytes(b"new audio")
        adapter.synthesize = AsyncMock(side_effect=synth)
        adapter.aclose = AsyncMock()
        first = asyncio.run(synthesize_edit(self.work, self.state["segments"][0], {}, {}, adapter))
        second = asyncio.run(synthesize_edit(self.work, self.state["segments"][0], {}, {}, adapter))
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual(first["text"], self.rows[0]["text"])
        self.assertTrue(resolve_replacement(self.work, first["path"]).is_file())
        with self.assertRaises(ArrangementError):
            resolve_replacement(self.work, "debug/_parsed_vtt.json")

    def test_preparation_reuses_user_audio_and_generates_only_changed_text(self):
        directory = self.work / "arrangement/audio"
        directory.mkdir(parents=True)
        (directory / "user.wav").write_bytes(b"user audio")
        (directory / "deferred.wav").write_bytes(b"deferred audio")
        self.state["segments"][0]["replacement"] = {"path": "arrangement/audio/user.wav", "text": self.rows[0]["text"]}
        self.state["segments"][1]["text"] = "之后生成的草稿"
        with patch("modules.arrangement.synthesize_edit", new=AsyncMock(return_value={"path": "arrangement/audio/deferred.wav", "text": "之后生成的草稿"})) as synth, patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 1000)):
            plan = prepare_arrangement_plan(self.work, Path("original.wav"), self.state, {}, {})
        synth.assert_awaited_once()
        self.assertEqual([clip.index for clip in plan.clips], [7, 13])
        self.assertEqual(plan.clips[0].path.name, "user.wav")
        self.assertEqual(self.state["segments"][1]["replacement"], None)

    def test_pipeline_passes_only_returned_plan_to_audio_adapter(self):
        plan = ArrangementPlan(Path("source.wav"), (), False)
        result = ArrangementAudioResult(object(), ())
        adapter = MagicMock()
        adapter.render.return_value = result
        payload = {"workDir": str(self.work), "audioPath": "source.wav", "state": self.state, "config": {}, "ttsKwargs": {}}
        with patch("modules.arrangement.prepare_arrangement_plan", return_value=plan), patch("modules.arrangement_audio.create_arrangement_audio_adapter", return_value=adapter), patch("modules.arrangement_audio.write_arrangement_result", return_value={"output_mp3": "result.mp3"}) as writer:
            output = run_arrangement_pipeline(payload)
        adapter.render.assert_called_once_with(plan)
        self.assertIs(writer.call_args.args[0], result)
        self.assertTrue(Path(output["log_path"]).is_file())

    def test_tts_readability_rules_are_shared_with_existing_optimization(self):
        for backend in ("edge-tts", "fish-audio", "minimax-tts"):
            prompt = build_phase2_system("", backend, "speech-2.8-hd")
            self.assertIn("重叠", prompt)
            self.assertIn("数字", prompt)
            self.assertIn("不改变片段边界", prompt)

    def test_japanese_regeneration_calls_service_each_time(self):
        from modules.vtt_parser import VTTBlock
        client = LLMClient(LLMConfig())
        client._post = AsyncMock(return_value="7\t请听我说话。")
        block = VTTBlock(7, 0, 1000, "旧中文")
        for _ in range(2):
            self.assertEqual(asyncio.run(client.optimize_from_japanese(block, "聞いてください")), "请听我说话。")
        self.assertEqual(client._post.await_count, 2)
        self.assertNotIn("旧中文", client._post.call_args.args[1])

    def test_gated_loudness_excludes_silence_and_handles_stereo_without_cancellation(self):
        from array import array
        buffer = MagicMock()
        buffer.frame_rate = 1000
        buffer.sample_width = 2
        buffer.channels = 2
        buffer.get_array_of_samples.return_value = array("h", [3277, -3277] * 20 + [0] * 80)
        self.assertAlmostEqual(perceived_db(buffer), -20, places=2)

    def test_render_preserves_gaps_and_uses_edited_loudness_without_mutating_source(self):
        from array import array
        from pydub import AudioSegment
        original = AudioSegment(array("h", [2000] * 4000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        tts = AudioSegment(array("h", [4000] * 1000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        plan = ArrangementPlan(Path("source.wav"), (
            ArrangedClip(7, 500, 1500, "第一段", Path("a.wav"), .5, 500, 1500),
            ArrangedClip(13, 2000, 3000, "第二段", Path("b.wav"), 0, 2000, 3000)), False)
        with patch("pydub.AudioSegment.from_file", side_effect=[original, tts, tts]):
            rendered = PydubArrangementAdapter().render(plan).audio
        self.assertEqual(len(rendered), 4000)
        self.assertEqual(rendered[0:400].rms, original[0:400].rms)
        self.assertAlmostEqual(rendered[600:1400].rms, 3000, delta=2)
        self.assertAlmostEqual(rendered[2100:2900].rms, 2000, delta=2)
        self.assertEqual(original.rms, 2000)

    def test_render_aligns_each_voice_to_its_own_original_interval(self):
        from array import array
        from pydub import AudioSegment
        original = AudioSegment(array("h", [1000] * 1000 + [4000] * 1000).tobytes(),
                                sample_width=2, frame_rate=1000, channels=1)
        tts = AudioSegment(array("h", [1000] * 1000).tobytes(),
                           sample_width=2, frame_rate=1000, channels=1)
        plan = ArrangementPlan(Path("source.wav"), (
            ArrangedClip(7, 0, 1000, "第一段", Path("a.wav"), 1, 0, 1000),
            ArrangedClip(13, 1000, 2000, "第二段", Path("b.wav"), 1, 1000, 2000)), False)
        with patch("pydub.AudioSegment.from_file", side_effect=[original, tts, tts]):
            rendered = PydubArrangementAdapter().render(plan).audio
        self.assertAlmostEqual(rendered[100:900].rms, 2000, delta=2)
        self.assertAlmostEqual(rendered[1100:1900].rms, 8000, delta=2)

    def test_render_uses_only_the_selected_tts_samples(self):
        from array import array
        from pydub import AudioSegment
        original = AudioSegment(array("h", [0] * 2000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        tts = AudioSegment(array("h", [0] * 500 + [4000] * 500).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        clip = ArrangedClip(7, 0, 500, "裁切", Path("tts.wav"), 1, 0, 500, 500, 1000)
        with patch("pydub.AudioSegment.from_file", side_effect=[original, tts]):
            rendered = PydubArrangementAdapter().render(ArrangementPlan(Path("source.wav"), (clip,), False)).audio
        self.assertGreater(rendered[0:450].rms, 0)
        self.assertEqual(rendered[550:1000].rms, 0)

    def test_render_keeps_the_complete_final_tts_after_the_original_ends(self):
        from array import array
        from pydub import AudioSegment
        original = AudioSegment(array("h", [2000] * 4000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        tts = AudioSegment(array("h", [4000] * 2000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        plan = ArrangementPlan(Path("source.wav"), (
            ArrangedClip(7, 3500, 5500, "末段", Path("last.wav"), 1, 3000, 4000),
        ), False)
        with patch("pydub.AudioSegment.from_file", side_effect=[original, tts]):
            rendered = PydubArrangementAdapter().render(plan).audio
        self.assertEqual(len(rendered), 5500)
        self.assertGreater(rendered[4500:5400].rms, 0)
        self.assertEqual(len(original), 4000)

    def test_first_voice_can_finish_five_seconds_before_source_and_export_subtitles_shift(self):
        from array import array
        from pydub import AudioSegment
        original = AudioSegment(array("h", [2000] * 4000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        tts = AudioSegment(array("h", [4000] * 2000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        plan = ArrangementPlan(Path("source.wav"), (
            ArrangedClip(7, -7000, -5000, "前置", Path("first.wav"), 1, 0, 1000),
        ), False)
        with patch("pydub.AudioSegment.from_file", side_effect=[original, tts]):
            result = PydubArrangementAdapter().render(plan)
        self.assertEqual(result.origin_shift_ms, 7000)
        self.assertEqual(len(result.audio), 11000)
        self.assertGreater(result.audio[0:2000].rms, 0)
        self.assertEqual(result.audio[2000:7000].rms, 0)
        self.assertEqual(result.audio[7000:11000].rms, original.rms)
        with patch("pydub.AudioSegment.export"):
            output = write_arrangement_result(result, self.work / "export", {})
        self.assertIn("00:00:00.000 --> 00:00:02.000", Path(output["final_vtt"]).read_text(encoding="utf-8"))

    def test_last_voice_can_start_five_seconds_after_source_but_no_later(self):
        from array import array
        from pydub import AudioSegment
        original = AudioSegment(array("h", [2000] * 4000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        tts = AudioSegment(array("h", [4000] * 2000).tobytes(), sample_width=2, frame_rate=1000, channels=1)
        clip = ArrangedClip(7, 9000, 11000, "后置", Path("last.wav"), 1, 3000, 4000)
        with patch("pydub.AudioSegment.from_file", side_effect=[original, tts]):
            rendered = PydubArrangementAdapter().render(ArrangementPlan(Path("source.wav"), (clip,), False)).audio
        self.assertEqual(len(rendered), 11000)
        self.assertEqual(rendered[4000:9000].rms, 0)
        self.assertGreater(rendered[9000:11000].rms, 0)
        with patch("pydub.AudioSegment.from_file", return_value=original), self.assertRaises(ArrangementError):
            PydubArrangementAdapter().render(ArrangementPlan(Path("source.wav"), (
                ArrangedClip(7, 9001, 11001, "过远", Path("last.wav"), 1, 3000, 4000),
            ), False))


if __name__ == "__main__":
    unittest.main()
