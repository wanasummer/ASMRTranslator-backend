import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from modules.arrangement.contracts import RenderedTake
from modules.arrangement.editor import run_editor_request
from modules.arrangement.plan import prepare_arrangement_plan
from modules.arrangement.state import initial_state, load_state, save_state
from modules.arrangement.takes import adopt_rendered_takes
from modules.storage import initialize_project, save_text


class RenderedTakeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        initialize_project(self.work)
        rows = [{"index": 7, "start_ms": 0, "end_ms": 1000, "text": "请听我说", "source_text": "聞いて"},
                {"index": 13, "start_ms": 2000, "end_ms": 3000, "text": "慢慢说吧", "source_text": "ゆっくり"}]
        save_text(self.work / "debug/_parsed_vtt.json", json.dumps(rows, ensure_ascii=False))
        audio = self.work / "arrangement/audio"
        audio.mkdir(parents=True)
        (audio / "take.wav").write_bytes(b"take")
        self.state = initial_state(self.work)
        merged, merged_away = self.state["segments"]
        merged.update(text="请听我说，慢慢说吧", interval={"start": 0, "end": 900}, ttsSample={"start": 0, "end": 900})
        merged_away["deleted"] = True
        save_state(self.work, self.state)
        self.take = RenderedTake(7, "请听我说，慢慢说吧", "arrangement/audio/take.wav", 0, 2600, 2600)

    def _save(self, state):
        return run_editor_request({"workDir": str(self.work), "action": "save", "state": state})["state"]

    def test_plan_reports_takes_synthesized_for_changed_text(self):
        with patch("modules.arrangement.plan.synthesize_edit", new=AsyncMock(return_value={"path": self.take.path, "text": self.take.text})), \
             patch("pydub.AudioSegment.from_file", return_value=MagicMock(__len__=lambda _: 2600)):
            plan = prepare_arrangement_plan(self.work, Path("original.wav"), self.state, {}, {})
        self.assertEqual(plan.takes, (self.take,))

    def test_adopted_take_becomes_the_fragment_audio(self):
        self.assertEqual(adopt_rendered_takes(self.work, [self.take]), 1)
        edit = load_state(self.work)["segments"][0]
        self.assertEqual(edit["replacement"], {"path": self.take.path, "text": self.take.text})
        self.assertEqual((edit["interval"], edit["ttsSample"]), ({"start": 0, "end": 2600}, {"start": 0, "end": 2600}))

    def test_take_is_skipped_when_the_text_changed_during_the_render(self):
        latest = load_state(self.work)
        latest["segments"][0]["text"] = "渲染期间又改了"
        save_state(self.work, latest)
        self.assertEqual(adopt_rendered_takes(self.work, [self.take]), 0)
        self.assertIsNone(load_state(self.work)["segments"][0]["replacement"])

    def test_take_keeps_its_fragment_profile_and_skips_a_changed_profile(self):
        profile = {"ttsBackend": "edge-tts", "edgeVoice": "a"}
        latest = load_state(self.work)
        latest["segments"][0]["tts"] = profile
        save_state(self.work, latest)
        self.assertEqual(adopt_rendered_takes(self.work, [self.take]), 0)
        tuned = RenderedTake(*[getattr(self.take, field) for field in ("index", "text", "path", "start_ms", "end_ms", "sample_end_ms")], profile)
        self.assertEqual(adopt_rendered_takes(self.work, [tuned]), 1)
        self.assertEqual(load_state(self.work)["segments"][0]["replacement"]["tts"], profile)

    def test_stale_studio_save_keeps_the_adopted_take_until_the_draft_is_loaded(self):
        adopt_rendered_takes(self.work, [self.take])
        stale = json.loads(json.dumps(self.state))
        stale["segments"][0]["volume"] = .5
        saved = self._save(stale)
        self.assertEqual((saved["segments"][0]["replacement"]["path"], saved["segments"][0]["volume"]), (self.take.path, .5))
        loaded = run_editor_request({"workDir": str(self.work), "action": "load"})["state"]
        loaded["segments"][0].update(replacement=None, interval=None, ttsSample=None)
        self.assertIsNone(self._save(loaded)["segments"][0]["replacement"])

    def test_studio_edit_of_the_audio_wins_over_the_adopted_take(self):
        adopt_rendered_takes(self.work, [self.take])
        edited = json.loads(json.dumps(self.state))
        edited["segments"][0]["interval"] = {"start": 100, "end": 1500}
        saved = self._save(edited)["segments"][0]
        self.assertEqual((saved["replacement"], saved["interval"]), (None, {"start": 100, "end": 1500}))


if __name__ == "__main__":
    unittest.main()
