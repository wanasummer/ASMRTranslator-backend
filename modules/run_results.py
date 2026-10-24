"""Concurrent, project-level run history. Subtitle text is rendered only on export."""
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .app_version import APP_VERSION


@contextmanager
def _locked(project: Path):
    with open(project / '.run-result.lock', 'a+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def record_run(output_dir: str, status: str, audio_path: str = '', asset_id: str = '', *,
               output_mode: str = 'audio') -> str:
    if output_mode not in {'audio', 'subtitles-only'}:
        raise ValueError('未知产物模式')
    work = Path(output_dir).resolve()
    project = work.parent
    target = project / 'run-result.json'
    with _locked(project):
        data = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'contractVersion': 2, 'runs': {}}
        record = {'status': status}
        if status == 'succeeded':
            source = Path(audio_path).resolve()
            stat = source.stat()
            mp3 = work / 'output.mp3'
            blocks = work / 'debug' / '_parsed_vtt.json'
            final_vtt = work / 'llm_subtitles' / 'final.vtt'
            if (not final_vtt.is_file() or not final_vtt.stat().st_size
                    or (output_mode == 'audio' and (
                        not mp3.is_file() or not mp3.stat().st_size or not blocks.is_file()))):
                raise ValueError('成功运行缺少音频或字幕记录')
            files = {'finalVtt': 'llm_subtitles/final.vtt'}
            if output_mode == 'audio':
                files.update({'mp3': 'output.mp3', 'blocks': 'debug/_parsed_vtt.json'})
                if (work / 'debug/high_quality_export.json').is_file():
                    files['highQualityRecipe'] = 'debug/high_quality_export.json'
            else:
                record['outputMode'] = output_mode
            record.update({
                'contractVersion': 1, 'appVersion': APP_VERSION,
                'assetId': asset_id or os.environ.get('ASMR_ASSET_ID', '').strip()
                or hashlib.sha256(str(source).casefold().encode('utf-8')).hexdigest()[:16],
                'source': {'audioPath': str(source), 'size': stat.st_size, 'modifiedAt': stat.st_mtime_ns // 1_000_000},
                'files': files,
                'completedAt': int(time.time() * 1000),
            })
        data['runs'][work.name] = record
        temporary = target.with_name(f'.run-result-{os.getpid()}.tmp')
        try:
            with open(temporary, 'w', encoding='utf-8', newline='\n') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return str(target)
