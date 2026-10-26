"""Player tags survive final exports without modifying source or audio samples."""
import base64
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
import json
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from mutagen import id3
from mutagen.flac import Picture
from mutagen.mp4 import MP4Cover
from mutagen.wave import WAVE

from modules.audio_metadata import (
    AudioMetadata, AudioMetadataError, MutagenMetadataReader,
    _convert_tags, write_audio_metadata,
)


def snapshot(tags, version=4):
    buffer = BytesIO()
    tags.save(buffer, v2_version=version, v23_sep=None)
    return AudioMetadata(buffer.getvalue(), version, len(tags), len(tags.getall("APIC")))


def write_wave(path):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x02" * 240)


class AudioMetadataTests(unittest.TestCase):
    def source_tags(self):
        tags = id3.ID3()
        tags.add(id3.TPE1(encoding=3, text=["作者一", "作者二", "声优三"]))
        tags.add(id3.TPE2(encoding=3, text=["社团名"]))
        tags.add(id3.TIT2(encoding=3, text=["原始标题"]))
        tags.add(id3.TALB(encoding=3, text=["专辑名"]))
        tags.add(id3.TCOP(encoding=3, text=["版权所有"]))
        tags.add(id3.TXXX(encoding=3, desc="自定义字段", text=["甲", "乙"]))
        tags.add(id3.PRIV(owner="example", data=b"\0\xffprivate"))
        tags.add(id3.COMM(encoding=3, lang="jpn", desc="备注", text=["原始备注"]))
        tags.add(id3.USLT(encoding=3, lang="jpn", desc="", text="原始歌词"))
        for description, kind in (("正面", 3), ("背面", 4)):
            tags.add(id3.APIC(encoding=3, mime="image/png", type=kind,
                              desc=description, data=b"\x89PNG" + bytes([kind])))
        return tags

    def test_mp3_and_wave_retain_all_frames_and_samples(self):
        source = self.source_tags()
        original = snapshot(source)
        with tempfile.TemporaryDirectory() as directory:
            mp3 = Path(directory) / "output.mp3"
            wav = Path(directory) / "output.wav"
            mp3.write_bytes(b"audio payload unchanged")
            write_wave(wav)
            report = write_audio_metadata(original, (str(mp3), str(wav)), directory)
            for copied in (id3.ID3(mp3), WAVE(wav).tags):
                self.assertEqual(set(copied), set(source))
                self.assertEqual(copied["TPE1"].text, ["作者一", "作者二", "声优三"])
                for key in source:
                    self.assertEqual(copied[key], source[key], key)
            self.assertTrue(mp3.read_bytes().endswith(b"audio payload unchanged"))
            with wave.open(str(wav), "rb") as audio:
                self.assertEqual(audio.readframes(240), b"\x01\x02" * 240)
            self.assertEqual(json.loads(Path(report).read_text(encoding="utf-8"))["picture_count"], 2)
            self.assertEqual((Path(directory) / "source_metadata.id3").read_bytes(), original.id3_bytes)
        self.assertEqual(snapshot(source), original)

    def test_id3_v23_keeps_multiple_authors(self):
        metadata = snapshot(self.source_tags(), version=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.mp3"
            path.write_bytes(b"audio")
            write_audio_metadata(metadata, (str(path),), directory)
            tags = id3.ID3(path, translate=False)
            self.assertEqual(tags.version[1], 3)
            self.assertEqual(tags["TPE1"].text, ["作者一", "作者二", "声优三"])

    def test_flac_keeps_repeated_authors_custom_tags_and_all_covers(self):
        picture = Picture()
        picture.type, picture.mime, picture.data = 3, "image/jpeg", b"jpeg bytes"
        audio = SimpleNamespace(tags={"ARTIST": ["甲", "乙"], "ALBUM": ["专辑"],
                                      "CUSTOM": ["值一", "值二"]}, pictures=[picture, picture])
        with patch("modules.audio_metadata.mutagen.File", return_value=audio):
            metadata = MutagenMetadataReader().read("source.flac")
        tags = id3.ID3(BytesIO(metadata.id3_bytes))
        self.assertEqual(tags["TPE1"].text, ["甲", "乙"])
        self.assertEqual(tags["TXXX:CUSTOM"].text, ["值一", "值二"])
        self.assertEqual(len(tags.getall("APIC")), 2)
        self.assertTrue(all(frame.data == picture.data for frame in tags.getall("APIC")))
        self.assertEqual(audio.tags["ARTIST"], ["甲", "乙"])

    def test_mp4_maps_people_track_and_embedded_cover(self):
        tags = _convert_tags(SimpleNamespace(tags={
            "©ART": ["甲", "乙"], "aART": ["社团"], "©wrt": ["作曲"],
            "trkn": [(2, 9)], "disk": [(1, 2)],
            "covr": [MP4Cover(b"png", imageformat=MP4Cover.FORMAT_PNG)],
            "----:com.apple.iTunes:CUSTOM": [b"custom text"],
        }))
        self.assertEqual(tags["TPE1"].text, ["甲", "乙"])
        self.assertEqual(tags["TRCK"].text, ["2/9"])
        self.assertEqual(tags["TPOS"].text, ["1/2"])
        self.assertEqual(tags.getall("APIC")[0].mime, "image/png")
        self.assertEqual(tags["TXXX:----:com.apple.iTunes:CUSTOM"].text, ["custom text"])

    def test_vorbis_embedded_picture(self):
        picture = Picture()
        picture.mime, picture.type, picture.data = "image/png", 3, b"picture"
        tags = _convert_tags(SimpleNamespace(tags={
            "metadata_block_picture": [base64.b64encode(picture.write()).decode("ascii")],
            "ARTIST": ["一", "二"],
        }))
        self.assertEqual(tags.getall("APIC")[0].data, b"picture")

    def test_wma_maps_author_and_picture(self):
        data = struct.pack("<BI", 3, 5) + "image/jpeg\0封面\0".encode("utf-16-le") + b"image"
        tags = _convert_tags(SimpleNamespace(tags={
            "Author": [SimpleNamespace(value="甲"), SimpleNamespace(value="乙")],
            "WM/Picture": [SimpleNamespace(value=data)],
        }))
        self.assertEqual(tags["TPE1"].text, ["甲", "乙"])
        self.assertEqual(tags.getall("APIC")[0].data, b"image")

    def test_wave_riff_info_and_no_metadata(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps({
            "format": {"tags": {"artist": "作者", "title": "标题", "custom": "内容"}}
        })))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            write_wave(path)
            before = path.read_bytes()
            metadata = MutagenMetadataReader(runner).read(str(path))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(id3.ID3(BytesIO(metadata.id3_bytes))["TPE1"].text, ["作者"])
            runner.return_value.stdout = "{}"
            empty = MutagenMetadataReader(runner).read(str(path))
            self.assertEqual(empty.frame_count, 0)
        self.assertIn("ffprobe", runner.call_args.args[0][0])

    def test_native_id3_reader_disables_frame_translation(self):
        audio = Mock(tags=self.source_tags())
        with patch("modules.audio_metadata.mutagen.File", return_value=audio):
            metadata = MutagenMetadataReader().read("source.mp3")
        self.assertEqual(metadata.picture_count, 2)
        audio.load.assert_called_once_with("source.mp3", translate=False)

    def test_real_wave_native_id3_reader_preserves_source(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0, stdout="{}"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            write_wave(path)
            audio = WAVE(path)
            audio.add_tags()
            audio.tags.update(self.source_tags())
            audio.save()
            before = path.read_bytes()
            metadata = MutagenMetadataReader(runner).read(str(path))
            self.assertEqual(metadata.picture_count, 2)
            self.assertEqual(id3.ID3(BytesIO(metadata.id3_bytes))["TPE1"].text,
                             ["作者一", "作者二", "声优三"])
            self.assertEqual(path.read_bytes(), before)

    def test_aac_leading_id3_preserves_custom_frames_and_cover(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.aac"
            path.write_bytes(snapshot(self.source_tags()).id3_bytes + b"aac payload")
            with patch("modules.audio_metadata.mutagen.File", return_value=None):
                metadata = MutagenMetadataReader().read(str(path))
            self.assertEqual(metadata.picture_count, 2)
            self.assertEqual(id3.ID3(BytesIO(metadata.id3_bytes))["TXXX:自定义字段"].text, ["甲", "乙"])

    def test_failures_are_domain_errors_and_do_not_replace_audio(self):
        with patch("modules.audio_metadata.mutagen.File", side_effect=ValueError("broken")):
            with self.assertRaises(AudioMetadataError):
                MutagenMetadataReader().read("broken.mp3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.wav"
            path.write_bytes(b"broken audio")
            with self.assertRaises(AudioMetadataError):
                write_audio_metadata(snapshot(self.source_tags()), (str(path),), directory)
            self.assertEqual(path.read_bytes(), b"broken audio")

    def test_pipeline_tags_both_outputs_before_recording_success(self):
        from modules.pipeline import run_pipeline
        from modules.vtt_parser import VTTBlock

        metadata = snapshot(self.source_tags())
        segment = {"index": 0, "tts_file": "tts.wav", "text": "这是测试字幕内容"}
        reader = Mock()
        reader.read.return_value = metadata
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output_dir = str(Path(directory) / "work")

            def join(segments, output, **kwargs):
                write_wave(output)
                Path(output).with_suffix(".mp3").write_bytes(b"audio")
                from modules.audio_joiner import JoinedAudio, SegmentPlacement
                return JoinedAudio(output, (SegmentPlacement("0", 0, 1000, 0, 1000),))

            def record(*args):
                self.assertEqual(id3.ID3(Path(output_dir) / "output.mp3")["TPE1"].text,
                                 ["作者一", "作者二", "声优三"])
                self.assertEqual(len(WAVE(Path(output_dir) / "output.wav").tags.getall("APIC")), 2)
                return "run-result.json"

            chain = Mock()
            chain.run.return_value = [segment]
            builder = Mock(subdir="built")
            builder.build.return_value = [segment]
            replacements = {
                "modules.pipeline.parse_vtt": [VTTBlock(0, 0, 1000, "这是测试字幕内容")],
                "modules.pipeline.get_audio_duration": 1,
                "modules.pipeline.split_audio": [segment],
                "modules.pipeline.generate_tts": [segment],
                "modules.pipeline.build_tts_handler_chain": chain,
                "modules.pipeline.align_tts_to_original": [segment],
                "modules.pipeline.enforce_tts_budget": [segment],
                "modules.pipeline.create_segment_builder": builder,
                "modules.audio_metadata.create_metadata_reader": reader,
                "modules.high_quality_export.write_export_recipe": "debug/high_quality_export.json",
            }
            for target, value in replacements.items():
                stack.enter_context(patch(target, return_value=value))
            stack.enter_context(patch("modules.pipeline.join_audio", side_effect=join))
            recorder = stack.enter_context(patch("modules.run_results.record_run", side_effect=record))
            writer = stack.enter_context(patch("modules.audio_metadata.write_audio_metadata",
                                               wraps=write_audio_metadata))
            result = run_pipeline("source.mp3", "subtitles.vtt", output_dir)
            reader.read.assert_called_once_with("source.mp3")
            self.assertIs(writer.call_args.args[0], metadata)
            self.assertEqual(writer.call_args.args[1], (result["output_mp3"], result["output_wav"]))
            self.assertTrue(Path(result["metadata_report"]).is_file())
            from modules.error import PipelineError
            writer.side_effect = AudioMetadataError("tagging failed")
            with self.assertRaises(PipelineError) as error:
                run_pipeline("source.mp3", "subtitles.vtt", output_dir)
            self.assertEqual(error.exception.descriptor.code, "AUDIO_METADATA_FAILED")
            recorder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
