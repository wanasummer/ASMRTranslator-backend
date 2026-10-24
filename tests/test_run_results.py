import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.run_results import record_run


class RunResultAssetIdentityTest(unittest.TestCase):
    def test_subtitles_only_record_requires_no_audio_or_tts_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            source.write_bytes(b"audio")
            work = Path(directory) / "subtitles"
            (work / "llm_subtitles").mkdir(parents=True)
            (work / "llm_subtitles" / "final.vtt").write_text("WEBVTT\n", encoding="utf-8")
            path = record_run(str(work), "succeeded", str(source), output_mode="subtitles-only")
            record = json.loads(Path(path).read_text(encoding="utf-8"))["runs"]["subtitles"]
            self.assertEqual(record["outputMode"], "subtitles-only")
            self.assertEqual(record["files"], {"finalVtt": "llm_subtitles/final.vtt"})
            with self.assertRaises(ValueError):
                record_run(str(work), "succeeded", str(source))
            (work / "llm_subtitles" / "final.vtt").unlink()
            with self.assertRaises(ValueError):
                record_run(str(work), "succeeded", str(source), output_mode="subtitles-only")

    def test_manifest_asset_id_is_used_instead_of_path_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source = project / "source.wav"
            source.write_bytes(b"audio")
            work = project / "stable-export"
            (work / "debug").mkdir(parents=True)
            (work / "llm_subtitles").mkdir()
            (work / "output.mp3").write_bytes(b"mp3")
            (work / "debug" / "_parsed_vtt.json").write_text("[]", encoding="utf-8")
            (work / "llm_subtitles" / "final.vtt").write_text("WEBVTT\n", encoding="utf-8")

            with patch.dict(os.environ, {"ASMR_ASSET_ID": "asset-stable-1"}):
                result_path = record_run(str(work), "succeeded", str(source))

            data = json.loads(Path(result_path).read_text(encoding="utf-8"))
            self.assertEqual(data["runs"]["stable-export"]["assetId"], "asset-stable-1")


if __name__ == "__main__":
    unittest.main()
