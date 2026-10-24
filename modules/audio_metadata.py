"""Read player metadata independently of audio processing, then tag final files.

ID3 frames (including private/custom frames and every picture) are retained as
ID3, without an EasyID3 whitelist. Other containers are translated to ID3;
unrecognized fields remain available as named TXXX frames. No audio is encoded.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Callable, Protocol

import mutagen
from mutagen import id3
from mutagen.flac import Picture
from mutagen.mp4 import MP4Cover
from mutagen.wave import WAVE

from .audio_splitter import get_ffmpeg_path
from .io import write_bytes, write_json


class AudioMetadataError(Exception):
    """Source metadata could not be read or final metadata could not be saved."""


@dataclass(frozen=True)
class AudioMetadata:
    """Owned ID3 snapshot; consumers never mutate the source tags or audio."""

    id3_bytes: bytes
    version: int
    frame_count: int
    picture_count: int


class MetadataReader(Protocol):
    def read(self, source: str) -> AudioMetadata: ...


_TEXT_FRAMES = {
    "title": "TIT2", "artist": "TPE1", "author": "TPE1",
    "album": "TALB", "albumartist": "TPE2", "album artist": "TPE2",
    "composer": "TCOM", "lyricist": "TEXT", "writer": "TEXT",
    "conductor": "TPE3", "remixer": "TPE4", "genre": "TCON",
    "date": "TDRC", "year": "TDRC", "copyright": "TCOP",
    "publisher": "TPUB", "organization": "TPUB", "isrc": "TSRC",
    "tracknumber": "TRCK", "track": "TRCK", "discnumber": "TPOS",
    "disc": "TPOS", "bpm": "TBPM", "encodedby": "TENC",
    "language": "TLAN", "artistsort": "TSOP", "albumsort": "TSOA",
    "titlesort": "TSOT", "albumartistsort": "TSO2", "composersort": "TSOC",
    "©nam": "TIT2", "©art": "TPE1", "aart": "TPE2",
    "©alb": "TALB", "©wrt": "TCOM", "©day": "TDRC", "©gen": "TCON",
    "cprt": "TCOP", "sonm": "TSOT", "soar": "TSOP", "soal": "TSOA",
    "soaa": "TSO2", "soco": "TSOC",
    "wm/albumtitle": "TALB", "wm/albumartist": "TPE2",
    "wm/composer": "TCOM", "wm/writer": "TEXT", "wm/genre": "TCON",
    "wm/year": "TDRC", "wm/tracknumber": "TRCK", "wm/partofset": "TPOS",
    "wm/publisher": "TPUB", "wm/isrc": "TSRC",
}


def _values(value) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        item = getattr(item, "value", item)
        if isinstance(item, bytes):
            try:
                item = item.decode("utf-8")
            except UnicodeDecodeError:
                item = "base64:" + base64.b64encode(item).decode("ascii")
        result.append(str(item))
    return result


def _add_text(tags: id3.ID3, key: str, values: list[str]) -> None:
    frame_name = _TEXT_FRAMES.get(key.lower())
    if frame_name:
        previous = tags.get(frame_name)
        # Preserve all people/values even when aliases occur in one container.
        text = list(previous.text) + values if previous else values
        tags.add(getattr(id3, frame_name)(encoding=3, text=text))
    elif key.lower() in {"comment", "description", "©cmt"}:
        tags.add(id3.COMM(encoding=3, lang="und", desc=key, text=values))
    elif key.lower() in {"lyrics", "unsyncedlyrics", "©lyr", "wm/lyrics"}:
        tags.add(id3.USLT(encoding=3, lang="und", desc=key, text="\n".join(values)))
    else:
        tags.add(id3.TXXX(encoding=3, desc=key, text=values))


def _add_picture(tags: id3.ID3, picture: Picture) -> None:
    # APIC identity is the description. Make duplicate descriptions unique so
    # front/back covers from FLAC/Vorbis are never overwritten.
    description = picture.desc
    suffix = 1
    while "APIC:" + description in tags:
        description = f"{picture.desc} ({suffix})"
        suffix += 1
    tags.add(id3.APIC(encoding=3, mime=picture.mime, type=picture.type,
                      desc=description, data=picture.data))


def _asf_picture(data: bytes) -> Picture:
    picture = Picture()
    picture.type, length = struct.unpack_from("<BI", data)
    offset = 5
    strings = []
    for _ in range(2):
        start = offset
        while data[offset:offset + 2] != b"\0\0":
            if offset + 2 > len(data):
                raise ValueError("truncated ASF picture string")
            offset += 2
        strings.append(data[start:offset].decode("utf-16-le"))
        offset += 2
    picture.mime, picture.desc = strings
    picture.data = data[offset:offset + length]
    if len(picture.data) != length:
        raise ValueError("truncated ASF picture")
    return picture


def _convert_tags(audio) -> id3.ID3:
    tags = id3.ID3()
    for key, value in (audio.tags or {}).items():
        lower = key.lower()
        if lower == "metadata_block_picture":
            for encoded in value:
                _add_picture(tags, Picture(base64.b64decode(encoded)))
        elif lower == "coverart":
            mime_types = (audio.tags or {}).get("coverartmime", [])
            for index, encoded in enumerate(value):
                picture = Picture()
                picture.type = 3
                picture.mime = mime_types[index] if index < len(mime_types) else "image/jpeg"
                picture.data = base64.b64decode(encoded)
                _add_picture(tags, picture)
        elif lower == "covr":
            for cover in value:
                picture = Picture()
                picture.type = 3
                picture.mime = "image/png" if cover.imageformat == MP4Cover.FORMAT_PNG else "image/jpeg"
                picture.data = bytes(cover)
                _add_picture(tags, picture)
        elif lower == "wm/picture":
            for attribute in value:
                _add_picture(tags, _asf_picture(attribute.value))
        elif lower in {"trkn", "disk"}:
            _add_text(tags, "tracknumber" if lower == "trkn" else "discnumber",
                      [f"{number}/{total}" if total else str(number) for number, total in value])
        else:
            _add_text(tags, key, _values(value))
    for picture in getattr(audio, "pictures", ()):
        _add_picture(tags, picture)
    return tags


class MutagenMetadataReader:
    def __init__(self, runner: Callable = subprocess.run):
        self._runner = runner

    def _probe_tags(self, source: str) -> dict:
        # WAV RIFF INFO and untagged AAC are not exposed by Mutagen. Use the
        # same explicitly bundled executable location as the audio decoder.
        ffmpeg = Path(get_ffmpeg_path())
        probe = str(ffmpeg.with_name("ffprobe.exe")) if ffmpeg.name == "ffmpeg.exe" else "ffprobe"
        result = self._runner(
            [probe, "-v", "error", "-show_entries", "format_tags:stream_tags",
             "-of", "json", source], capture_output=True, text=True,
            encoding="utf-8", timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise AudioMetadataError("ffprobe could not read source metadata")
        data = json.loads(result.stdout)
        tags = dict(data.get("format", {}).get("tags", {}))
        for stream in data.get("streams", []):
            for key, value in stream.get("tags", {}).items():
                tags.setdefault(key, value)
        return tags

    def read(self, source: str) -> AudioMetadata:
        try:
            audio = mutagen.File(source, easy=False)
            if audio is not None and isinstance(audio.tags, id3.ID3):
                # Disable automatic ID3 upgrades/removal of obsolete frames.
                audio.load(source, translate=False)
                tags = audio.tags
            else:
                tags = _convert_tags(audio) if audio is not None else id3.ID3()
                if audio is None or audio.tags is None:
                    # Raw AAC can carry a leading ID3 tag even though the AAC
                    # adapter itself exposes no tags.
                    try:
                        tags = id3.ID3(source, translate=False)
                    except id3.ID3NoHeaderError:
                        pass
            if (not tags and (audio is None or audio.tags is None)) or isinstance(audio, WAVE):
                for key, value in self._probe_tags(source).items():
                    # Do not duplicate native ID3 fields projected by ffprobe.
                    if _TEXT_FRAMES.get(key.lower()) not in tags:
                        _add_text(tags, key, _values(value))
            version = tags.version[1] if tags.version[1] in (3, 4) else 4
            buffer = BytesIO()
            # Container-specific ID3 subclasses save a whole WAV/AIFF file;
            # the snapshot is a standalone ID3 block, so use the base writer.
            id3.ID3.save(tags, buffer, v2_version=version, v23_sep=None)
            return AudioMetadata(buffer.getvalue(), version, len(tags), len(tags.getall("APIC")))
        except AudioMetadataError:
            raise
        except Exception as error:
            raise AudioMetadataError("could not read source audio metadata") from error


def create_metadata_reader() -> MetadataReader:
    """Factory for the player-metadata adapter, independent of orchestration."""
    return MutagenMetadataReader()


def write_audio_metadata(metadata: AudioMetadata, outputs: tuple[str, ...],
                         audit_dir: str) -> str:
    """Persist the source snapshot before tagging MP3/WAV copies atomically.

    Output audio bytes are copied, never decoded/re-encoded. Source snapshots
    also retain fields that a particular player may not display.
    """
    try:
        raw_path = Path(audit_dir) / "source_metadata.id3"
        write_bytes(str(raw_path), metadata.id3_bytes)
        written = []
        for output in outputs:
            path = Path(output)
            if path.suffix.lower() not in {".mp3", ".wav"}:
                raise AudioMetadataError(f"unsupported metadata output: {path.suffix}")
            with tempfile.TemporaryDirectory(prefix=".metadata-", dir=path.parent) as temporary:
                staged = Path(temporary) / path.name
                shutil.copyfile(path, staged)
                tags = id3.ID3(BytesIO(metadata.id3_bytes), translate=False)
                if path.suffix.lower() == ".wav":
                    audio = WAVE(staged)
                    if audio.tags is None:
                        audio.add_tags()
                    audio.tags.clear()
                    audio.tags.update(tags)
                    audio.tags.unknown_frames = list(tags.unknown_frames)
                    audio.save(v2_version=metadata.version, v23_sep=None)
                else:
                    tags.save(staged, v2_version=metadata.version, v23_sep=None)
                staged.replace(path)
            written.append(str(path))
        report = str(Path(audit_dir) / "metadata_copy.json")
        write_json(report, {"source_snapshot": str(raw_path), "frame_count": metadata.frame_count,
                            "picture_count": metadata.picture_count, "outputs": written,
                            "status": "copied"})
        print(f"       [audio.metadata] {metadata.frame_count} tags, {metadata.picture_count} pictures → {report}")
        return report
    except AudioMetadataError:
        raise
    except Exception as error:
        raise AudioMetadataError("could not persist final audio metadata") from error
