from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioTask:
    source: Path
    output_base: Path
    missing_formats: tuple[str, ...]


OUTPUT_EXTENSIONS = {
    "srt": "srt",
    "vtt": "vtt",
    "lrc": "lrc",
    "txt": "txt",
    "json": "json",
    "json-full": "json",
    "json_full": "json",
    "csv": "csv",
}


def discover_audio(paths: list[str], suffixes: list[str], output_dir: str | None, formats: list[str], overwrite: bool) -> list[AudioTask]:
    tasks: list[AudioTask] = []
    normalized_suffixes = {suffix.lower().lstrip(".") for suffix in suffixes}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            root_parent = path.parent
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower().lstrip(".") in normalized_suffixes:
                    task = _make_task(child, child.relative_to(root_parent).with_suffix(""), output_dir, formats, overwrite)
                    if task is not None:
                        tasks.append(task)
        elif path.is_file() and path.suffix.lower().lstrip(".") in normalized_suffixes:
            task = _make_task(path, Path(path.name).with_suffix(""), output_dir, formats, overwrite)
            if task is not None:
                tasks.append(task)
    return tasks


def _make_task(source: Path, relative_base: Path, output_dir: str | None, formats: list[str], overwrite: bool) -> AudioTask | None:
    if output_dir:
        output_base = Path(output_dir).expanduser() / relative_base
    else:
        output_base = source.with_suffix("")
    missing = tuple(
        fmt
        for fmt in formats
        if overwrite or not output_base.with_suffix(f".{OUTPUT_EXTENSIONS.get(fmt, fmt)}").exists()
    )
    if not missing:
        return None
    return AudioTask(source=source, output_base=output_base, missing_formats=missing)
