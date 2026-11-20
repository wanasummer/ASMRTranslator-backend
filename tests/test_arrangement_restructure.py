from modules.storage import initialize_project, save_text, address
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from modules.arrangement.editor import run_editor_request
from modules.arrangement.restructure import FragmentRow, apply_restructure_plan, restructure_rows
from modules.arrangement.state import initial_state, load_state, publish_subtitle_base, save_state
from modules.error import LlmError, descriptor
from modules.llm import LLMConfig
from modules.llm.capabilities.capRestructure.parsing import parse_restructure_plan
from modules.llm.contracts import RestructureOperation, RestructurePlan
from modules.llm.factory import create_capabilities
from modules.subtitle_processing.editor import SubtitleEditor


def _rows(*indices):
    return [FragmentRow(i, f"中文{i}", f"日本語{i}", None, i * 1000, i * 1000 + 800) for i in indices]


class RestructureParsingTests(unittest.TestCase):
    def test_valid_operations_survive_and_invalid_ones_are_reported(self):
        raw = "```json\n" + json.dumps({"operations": [
            {"type": "modify", "index": 1, "text": "改好的中文", "reason": "错字"},
            {"type": "delete", "index": 2},
            {"type": "merge", "indices": [3, 4], "text": "合并后的句子"},
            {"type": "merge", "indices": [5, 7], "text": "不相邻"},
            {"type": "delete", "index": 3},
            {"type": "modify", "index": 99, "text": "不在本批"},
            {"type": "modify", "index": 6, "text": "  "},
            {"type": "split", "index": 6},
        ]}, ensure_ascii=False) + "\n```"
        plan = parse_restructure_plan(raw, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(plan.operations, (
            RestructureOperation("modify", (1,), "改好的中文", "错字"),
            RestructureOperation("delete", (2,)),
            RestructureOperation("merge", (3, 4), "合并后的句子")))
        self.assertEqual(len(plan.rejected), 5)

    def test_unusable_response_raises(self):
        for raw in ("没有改动", '{"ops": []}', "[1, 2]"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_restructure_plan(raw, [1])


class RestructureCapabilityTests(unittest.TestCase):
    def test_prompt_carries_summary_instruction_and_both_languages(self):
        capability = create_capabilities(LLMConfig()).restructure
        capability.runtime.complete = AsyncMock(return_value='{"operations":[{"type":"delete","index":2}]}')
        plan = asyncio.run(capability.plan(_rows(1, 2), "剧情摘要", "删除语气词"))
        self.assertEqual(plan.operations, (RestructureOperation("delete", (2,)),))
        system, user = capability.runtime.complete.call_args.args[:2]
        self.assertIn("剧情摘要", system)
        self.assertIn("删除语气词", system)
        self.assertEqual(user.splitlines()[0], "1\t1000-1800\t-\t日本語1\t中文1")

    def test_invalid_format_is_retried_once_then_raises_typed_error(self):
        capability = create_capabilities(LLMConfig()).restructure
        capability.runtime.complete = AsyncMock(side_effect=["不是 JSON", '{"operations":[]}'])
        self.assertEqual(asyncio.run(capability.plan(_rows(1), "", "要求")).operations, ())
        capability.runtime.complete = AsyncMock(return_value="不是 JSON")
        with self.assertRaises(LlmError):
            asyncio.run(capability.plan(_rows(1), "", "要求"))
        self.assertEqual(capability.runtime.complete.await_count, 2)


class SubtitleEditorRestructureTests(unittest.TestCase):
    def test_editor_reuses_the_summary_capability_and_passes_it_to_planning(self):
        capabilities = create_capabilities(LLMConfig())
        capabilities.summary.summarize = AsyncMock(return_value="整体情节")
        capabilities.restructure.plan = AsyncMock(return_value=RestructurePlan(()))
        editor = SubtitleEditor(capabilities)
        rows = _rows(1, 2)
        summary = asyncio.run(editor.summarize(rows))
        asyncio.run(editor.restructure(rows, summary, "要求"))
        capabilities.summary.summarize.assert_awaited_once_with(rows)
        capabilities.restructure.plan.assert_awaited_once_with(rows, "整体情节", "要求")


class ApplyRestructurePlanTests(unittest.TestCase):
    def setUp(self):
        self.state = {"schemaVersion": 1, "sourceVersion": "v", "loudnessScope": "segment", "globalVolume": 1,
                      "segments": [{"index": i, "text": f"中文{i}", "japanese": f"日{i}", "volume": 1,
                                    "interval": {"start": i * 1000, "end": i * 1000 + 500} if i == 3 else None,
                                    "replacement": None, "deleted": i == 5} for i in range(1, 7)]}

    def test_merge_keeps_first_fragment_and_audio_is_untouched(self):
        plan = RestructurePlan((RestructureOperation("modify", (1,), "改"),
                                RestructureOperation("delete", (2,)),
                                RestructureOperation("merge", (3, 4, 6), "合并")))
        result, report, changes = apply_restructure_plan(self.state, plan)
        edits = {edit["index"]: edit for edit in result["segments"]}
        self.assertEqual([change["index"] for change in changes], [1, 2, 3, 4, 6])
        self.assertEqual(changes[2], {"index": 3, "text": "合并", "japanese": "日3日4日6", "deleted": False, "blockId": None})
        self.assertEqual(edits[1]["text"], "改")
        self.assertTrue(edits[2]["deleted"])
        self.assertEqual((edits[3]["text"], edits[3]["japanese"]), ("合并", "日3日4日6"))
        self.assertEqual(edits[3]["interval"], {"start": 3000, "end": 3500})
        self.assertTrue(edits[4]["deleted"] and edits[6]["deleted"])
        self.assertEqual(report, {"modified": 1, "deleted": 1, "merged": 1, "mergedFragments": 3, "skipped": []})
        self.assertEqual(self.state["segments"][0]["text"], "中文1")

    def test_merge_across_blocks_moves_every_fragment_of_the_later_block_into_the_first(self):
        blocks = {1: "a", 2: "a", 3: "b", 4: "b", 5: "c", 6: "c"}
        plan = RestructurePlan((RestructureOperation("merge", (2, 3), "跨块合并"),))
        result, _, changes = apply_restructure_plan(self.state, plan, blocks=blocks)
        self.assertEqual({edit["index"]: edit.get("blockId") for edit in result["segments"]},
                         {1: None, 2: None, 3: "a", 4: "a", 5: None, 6: None})
        self.assertEqual([(change["index"], change["blockId"]) for change in changes], [(2, None), (3, "a"), (4, "a")])

    def test_merge_inside_one_block_keeps_blocks(self):
        plan = RestructurePlan((RestructureOperation("merge", (3, 4), "同块合并"),))
        result, _, changes = apply_restructure_plan(self.state, plan, blocks={1: "a", 2: "a", 3: "b", 4: "b", 5: "c", 6: "c"})
        self.assertFalse(any("blockId" in edit for edit in result["segments"]))
        self.assertEqual([change["index"] for change in changes], [3, 4])

    def test_merge_over_a_deleted_block_joins_everything_it_spans(self):
        blocks = {1: "a", 2: "a", 3: "b", 4: "b", 5: "c", 6: "d"}
        plan = RestructurePlan((RestructureOperation("merge", (4, 6), "跨过已删除"),))
        result, _, _ = apply_restructure_plan(self.state, plan, blocks=blocks)
        self.assertEqual({edit["index"]: edit.get("blockId") for edit in result["segments"]},
                         {1: None, 2: None, 3: None, 4: None, 5: "b", 6: "b"})

    def test_fragments_changed_since_planning_are_skipped_whole(self):
        latest = json.loads(json.dumps(self.state))
        latest["segments"][3]["text"] = "用户随后改过"
        plan = RestructurePlan((RestructureOperation("merge", (3, 4), "合并"),
                                RestructureOperation("delete", (5,))), ("模型无效操作",))
        result, report, changes = apply_restructure_plan(latest, plan, self.state)
        self.assertEqual(result["segments"], latest["segments"])
        self.assertEqual(changes, [])
        self.assertEqual(len(report["skipped"]), 3)

    def test_rows_expose_only_active_fragments_with_source_timing(self):
        aligned = [(i, {"index": e["index"], "start_ms": i * 10, "end_ms": i * 10 + 5, "speaker": "A"})
                   for i, e in enumerate(self.state["segments"])]
        rows = restructure_rows(self.state, aligned)
        self.assertEqual([row.index for row in rows], [1, 2, 3, 4, 6])
        self.assertEqual((rows[0].source_text, rows[0].start_ms, rows[0].speaker), ("日1", 0, "A"))


class RestructureEditorRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        initialize_project(self.work)
        (self.work / "debug").mkdir()
        rows = [{"index": i, "start_ms": i * 1000, "end_ms": i * 1000 + 800, "text": f"中文{i}", "source_text": f"日{i}"}
                for i in range(3)]
        save_text(self.work / "debug/_parsed_vtt.json", json.dumps(rows, ensure_ascii=False))
        self.state = initial_state(self.work)
        save_state(self.work, self.state)

    def _request(self, action, **values):
        return {"workDir": str(self.work), "action": action, "state": self.state, "config": {}, **values}

    def test_summary_reads_active_rows_and_degrades_to_empty_on_failure(self):
        self.state["segments"][2]["deleted"] = True
        client = MagicMock()
        client.summarize = AsyncMock(return_value="情节")
        with patch("modules.arrangement.optimization.create_optimizer", return_value=client):
            self.assertEqual(run_editor_request(self._request("summarize-arrangement"))["summary"], "情节")
            self.assertEqual([row.index for row in client.summarize.call_args.args[0]], [0, 1])
            client.summarize = AsyncMock(side_effect=LlmError(descriptor("LLM_SUMMARY_FAILED")))
            result = run_editor_request(self._request("summarize-arrangement"))
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["summaryError"]["code"], "LLM_SUMMARY_FAILED")

    def test_batch_planner_receives_only_its_rows_and_changes_are_saved_with_audit(self):
        client = MagicMock()
        client.restructure = AsyncMock(return_value=RestructurePlan((RestructureOperation("merge", (0, 1), "合并句"),)))
        payload = self._request("optimize-arrangement", instruction="合并切分错误", summary="情节", indices=[0, 1])
        with patch("modules.arrangement.optimization.create_optimizer", return_value=client):
            result = run_editor_request(payload)
        rows, summary, instruction = client.restructure.call_args.args
        self.assertEqual(([row.index for row in rows], summary, instruction), ([0, 1], "情节", "合并切分错误"))
        self.assertEqual(result["changes"], [{"index": 0, "text": "合并句", "japanese": "日0日1", "deleted": False, "blockId": None},
                                             {"index": 1, "text": "中文1", "japanese": "日1", "deleted": True, "blockId": "row-0"}])
        saved = load_state(self.work)
        self.assertEqual((saved["segments"][0]["text"], saved["segments"][1]["deleted"], saved["segments"][2]["text"]),
                         ("合并句", True, "中文2"))
        self.assertEqual([edit.get("blockId") for edit in saved["segments"]], [None, "row-0", None])
        audit = address(self.work / "arrangement/audit").store.keys("", "arrangement/audit/")
        self.assertEqual(len(audit), 2)

    def test_single_subtitle_base_is_replaced_by_each_published_arrangement(self):
        base = lambda: run_editor_request(self._request("subtitle-base"))["texts"]
        self.assertEqual(base(), {"0": "中文0", "1": "中文1", "2": "中文2"})
        published = json.loads(json.dumps(self.state))
        published["segments"][1]["text"] = "第一次编排"
        publish_subtitle_base(self.work, published)
        published["segments"][1]["text"] = "第二次编排"
        publish_subtitle_base(self.work, published)
        self.assertEqual(base(), {"0": "中文0", "1": "第二次编排", "2": "中文2"})
        self.assertEqual(address(self.work / "arrangement").store.keys("", "arrangement/subtitle_base"),
                         ["arrangement/subtitle_base.json"])
        publish_subtitle_base(self.work, {**published, "sourceVersion": "reconverted"})
        self.assertEqual(base()["1"], "中文1")

    def test_batch_with_deleted_fragment_is_rejected(self):
        self.state["segments"][1]["deleted"] = True
        payload = self._request("optimize-arrangement", instruction="要求", summary="", indices=[0, 1])
        with patch("modules.arrangement.optimization.create_optimizer") as factory, self.assertRaises(Exception):
            run_editor_request(payload)
        factory.assert_not_called()

    def test_empty_instruction_is_rejected_before_any_request(self):
        client = MagicMock()
        payload = self._request("optimize-arrangement", instruction="  ", summary="", indices=[0])
        with patch("modules.arrangement.optimization.create_optimizer", return_value=client), self.assertRaises(Exception):
            run_editor_request(payload)
        client.restructure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
