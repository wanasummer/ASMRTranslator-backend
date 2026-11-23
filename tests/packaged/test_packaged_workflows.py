"""Core end-to-end packaged sidecar tests covering >90% of real user workflows."""

from __future__ import annotations

import json
import re
import wave
from mutagen import id3

from modules.storage import contains, read_text
from .harness import PackagedSidecarTestCase, RecordedRequest


class PackagedWorkflowTests(PackagedSidecarTestCase):
    def test_native_vad_and_japanese_mecab_runtimes_in_clean_room(self) -> None:
        for flag in ("--check-vad-runtime", "--check-japanese-runtime"):
            res = self.run_sidecar([flag], inject_env=False)
            payload = json.loads(res.stdout.strip())
            self.assertTrue(payload.get("available"), f"{flag} unavailable: {payload}")

    def test_preflight_protocol_and_custom_stub_route_extensibility(self) -> None:
        config = {
            "llmProvider": "openai",
            "llmBaseUrl": f"{self.stub.base_url}/v1",
            "llmApiKey": "cleanroom-llm-key",
            "llmModel": "gpt-4o-mini",
            "asrEngine": "common-asr",
            "asrBaseUrl": f"{self.stub.base_url}/asr",
            "asrPort": self.stub.port,
            "ttsBackend": "common-tts",
            "commonTtsManifest": self.common_tts_manifest(),
            "commonTtsBaseUrl": f"{self.stub.base_url}/tts",
            "commonTtsApiKey": "cleanroom-tts-key",
            "commonTtsParameters": {},
        }
        for service in ("llm", "asr", "tts"):
            res = self.run_sidecar(
                ["--preflight"],
                stdin={"service": service, "config": config},
                inject_env=True,
            )
            payload = json.loads(res.stdout.strip())
            self.assertTrue(payload.get("success"), f"Preflight {service} failed: {payload}")

        # Verify custom route override works for future feature tests
        def custom_asr_health(_req: RecordedRequest):
            return 503, {}, {"status": "busy", "ready": False}

        self.stub.add_route("GET", "/asr/health", custom_asr_health)
        res = self.run_sidecar(
            ["--preflight"],
            stdin={"service": "asr", "config": config},
            inject_env=True,
        )
        payload = json.loads(res.stdout.strip())
        self.assertFalse(payload.get("success"))

    def test_subtitle_phantom_stretch_arrangement_and_high_quality_export(self) -> None:
        fixture = self.create_user_project()
        out_dir = fixture.project_dir / "generated" / "assets" / "01導入_A"

        # 1. Main pipeline with UTF-16 LRC + Phantom mode + follow_original timeline render
        #    Run with inject_env=False so PATH only has System32 and ASMR_FFMPEG_DIR is unset.
        res = self.run_sidecar(
            [
                str(fixture.source_mp3),
                str(fixture.source_lrc),
                "--output",
                str(out_dir),
                "--phantom-mode",
                "--sync-mode",
                "follow_original",
                "--llm-kwargs",
                json.dumps(self.llm_kwargs(), ensure_ascii=False),
                "--tts-backend",
                "common-tts",
                "--tts-backend-kwargs-stdin",
            ],
            stdin=self.common_tts_kwargs(),
            inject_env=False,
        )
        result = self.extract_pipeline_result(res.stdout)
        self.assertTrue(result.get("success"))
        self.assertRegex(res.stdout, r"\[arrangement\] 片段 0:.*语速 (?!1\.00×)\d+\.\d+×")

        mp3_path = out_dir / "output.mp3"
        wav_path = out_dir / "output.wav"
        db_path = fixture.project_dir / "generated" / "project.sqlite3"
        self.assertTrue(mp3_path.is_file() and mp3_path.stat().st_size > 1000)
        self.assertTrue(wav_path.is_file() and wav_path.stat().st_size > 1000)
        self.assertTrue(db_path.is_file())

        # Deep artifact check: ID3 tags & cover picture preserved on exported MP3
        exported_tags = id3.ID3(mp3_path)
        self.assertEqual(exported_tags["TIT2"].text, ["01導入"])
        self.assertEqual(exported_tags["TPE1"].text, ["テスト声優一", "テスト声優二"])
        self.assertEqual(len(exported_tags.getall("APIC")), 1)

        # Deep artifact check: WAV preserves source 44.1kHz stereo and SQLite has final.vtt
        with wave.open(str(wav_path), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 2)
            self.assertEqual(wf.getframerate(), 44100)
            self.assertGreater(wf.getnframes(), 44100)
        self.assertTrue(contains(out_dir / "llm_subtitles/final.vtt"))
        self.assertIn("WEBVTT", read_text(out_dir / "llm_subtitles/final.vtt"))

        # 2. Studio Arrangement Editor (--arrangement-editor load -> edit -> save -> --arrangement-run)
        load_res = self.run_sidecar(
            ["--arrangement-editor"],
            stdin={"workDir": str(out_dir), "action": "load"},
            inject_env=True,
        )
        loaded = json.loads(load_res.stdout.strip())
        self.assertTrue(loaded.get("success"))
        state = loaded["state"]
        state["segments"][0]["interval"] = {"start": 200, "end": 700}

        save_res = self.run_sidecar(
            ["--arrangement-editor"],
            stdin={"workDir": str(out_dir), "action": "save", "state": state},
            inject_env=True,
        )
        saved = json.loads(save_res.stdout.strip())
        self.assertTrue(saved.get("success"))

        run_res = self.run_sidecar(
            ["--arrangement-run"],
            stdin={
                "workDir": str(out_dir),
                "audioPath": str(fixture.source_mp3),
                "state": saved["state"],
                "config": {"ttsBackend": "common-tts"},
                "ttsKwargs": self.common_tts_kwargs(),
            },
            inject_env=False,
        )
        arrange_result = self.extract_pipeline_result(run_res.stdout)
        self.assertTrue(arrange_result.get("success"))

        # 3. High-Quality Export (--export-high-quality)
        hq_mp3 = self.temp_root / "hq_export_01導入.mp3"
        recipe_path = out_dir / "debug" / "high_quality_export.json"
        self.run_sidecar(
            ["--export-high-quality", str(recipe_path), "--output", str(hq_mp3)],
            inject_env=False,
        )
        self.assertTrue(hq_mp3.is_file() and hq_mp3.stat().st_size > 1000)

    def test_no_subtitle_native_vad_asr_mecab_cleaning_and_dual_mode(self) -> None:
        fixture = self.create_user_project()
        out_dir = fixture.project_dir / "generated" / "assets" / "01導入_B"

        res = self.run_sidecar(
            [
                str(fixture.source_mp3),
                "--output",
                str(out_dir),
                "--retain-original",
                "--sync-mode",
                "none",
                "--asr-backend",
                "common-asr",
                "--asr-kwargs",
                json.dumps({"endpoint": f"{self.stub.base_url}/asr", "output_language": "ja"}),
                "--llm-kwargs",
                json.dumps(self.llm_kwargs(), ensure_ascii=False),
                "--tts-backend",
                "common-tts",
                "--tts-backend-kwargs-stdin",
            ],
            stdin=self.common_tts_kwargs(),
            inject_env=True,
        )
        result = self.extract_pipeline_result(res.stdout)
        self.assertTrue(result.get("success"))
        self.assertTrue((out_dir / "output.mp3").is_file())
        self.assertTrue((out_dir / "output.wav").is_file())
        self.assertTrue(any(r.path == "/asr/job" for r in self.stub.requests))


if __name__ == "__main__":
    unittest.main()
