"""Offline fixtures and stage invocation helpers for the LLM module tests."""

from types import SimpleNamespace
from modules.subtitle_processing import create_subtitle_stage, SubtitleStageInput, SubtitleStageResult


def process_subtitles(blocks, settings=None, *, output_dir=None):
    result = create_subtitle_stage(settings or {}, output_dir).run(SubtitleStageInput(tuple(blocks)))
    return list(result.blocks), result.stats


def process_asr(blocks, settings=None, *, preserve_timeline=False,
                preserve_source_timeline=False, output_dir=None):
    result = create_subtitle_stage(settings or {}, output_dir).run(SubtitleStageInput(
        tuple(blocks), source="asr", preserve_timeline=preserve_timeline,
        preserve_source_timeline=preserve_source_timeline,
    ))
    return list(result.blocks), result.stats


def subtitle_stage_factory(transform):
    def create(settings, output_dir=None):
        def run(value):
            blocks, stats = transform(list(value.blocks), settings, output_dir=output_dir,
                                     preserve_source_timeline=value.preserve_source_timeline)
            return SubtitleStageResult(tuple(blocks), stats)
        return SimpleNamespace(run=run)
    return create


class FakeResponse:
    def __init__(self, status_code, json_data, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


class FakeAsyncContext:
    def __init__(self, owner):
        self.owner = owner

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers, json):
        self.owner.calls.append({"url": url, "headers": headers, "json": json})
        if self.owner.responses:
            return self.owner.responses.pop(0)
        return FakeResponse(500, {}, "no response queued")


class FakeHttpx:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def AsyncClient(self, **kwargs):
        return FakeAsyncContext(self)
