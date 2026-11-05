"""Explicit run lifecycle records; exports never infer success from logs."""
import os
from pathlib import Path
import time
from uuid import uuid4
from .app_version import APP_VERSION
from .storage import address, contains, encode_json, StorageError


def record_arrangement_run(work: Path, output: Path, snapshot: dict, files: dict) -> None:
    """Publish only the successfully persisted audio and subtitle snapshot."""
    target = address(work / 'run')
    record = {'snapshot': snapshot, 'directory': output.relative_to(work).as_posix(), 'files': files}
    with target.store.connect() as connection:
        connection.execute('INSERT INTO runs(id,scope,kind,status,payload,updated_at) VALUES(?,?,?,?,?,?)',
                           (output.name, target.scope, 'arrangement', 'succeeded',
                            encode_json(record).decode('utf-8'), time.time_ns() // 1_000_000))


def record_run(output_dir: str, status: str, audio_path: str = '', asset_id: str = '', *,
               output_mode: str = 'audio') -> str:
    if output_mode not in {'audio', 'subtitles-only'}:
        raise ValueError('未知产物模式')
    work = Path(output_dir).resolve()
    target = address(work / 'run')
    record = {'status': status, 'outputMode': output_mode}
    if status == 'succeeded':
        source = Path(audio_path).resolve()
        stat = source.stat()
        if (not contains(work / 'llm_subtitles/final.vtt') or
            (output_mode == 'audio' and (not (work / 'output.mp3').is_file()
                                        or not (work / 'output.mp3').stat().st_size
                                        or not contains(work / 'debug/_parsed_vtt.json')))):
            raise StorageError('成功运行缺少音频或字幕记录')
        files = {'finalVtt': 'llm_subtitles/final.vtt'}
        if output_mode == 'audio':
            files.update({'mp3': 'output.mp3', 'blocks': 'debug/_parsed_vtt.json'})
            if contains(work / 'debug/high_quality_export.json'):
                files['highQualityRecipe'] = 'debug/high_quality_export.json'
        record.update({
            'contractVersion': 1, 'appVersion': APP_VERSION,
            'assetId': asset_id or os.environ.get('ASMR_ASSET_ID', '') or target.scope or work.name,
            'source': {'audioPath': str(source), 'size': stat.st_size,
                       'modifiedAt': stat.st_mtime_ns // 1_000_000},
            'files': files, 'completedAt': time.time_ns() // 1_000_000,
        })
        project_root = target.store.root.parent if target.store.root.name == 'generated' else target.store.root
        if source.is_relative_to(project_root):
            record['source']['projectRelativePath'] = source.relative_to(project_root).as_posix()
    with target.store.connect() as connection:
        connection.execute('INSERT INTO runs(id,scope,kind,status,payload,updated_at) VALUES(?,?,?,?,?,?)',
                           (uuid4().hex, target.scope, 'pipeline', status, encode_json(record).decode('utf-8'),
                            time.time_ns() // 1_000_000))
    return target.reference
