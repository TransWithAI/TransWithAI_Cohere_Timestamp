from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


BUILTIN_DEFAULTS: dict[str, Any] = {
    "language": "ja",
    "sub_formats": ["srt"],
    "audio_suffixes": ["wav", "flac", "mp3"],
    "model_name_or_path": "models/cohere/model.gguf",
    "qwen3_forced_aligner_path": "models/qwen3-forced-aligner.gguf",
    "force_aligner": True,
    "crispasr_path": "auto",
    "device": "auto",
    "compute_type": "auto",
    "gpu_backend": None,
    "vad_filter": True,
    "vad_model": "models/vad/whisper-vad-asmr-q4_k.gguf",
    "split_on_punct": True,
    "output_dir": None,
    "overwrite": False,
    "log_level": "DEBUG",
    "dry_run": False,
    "segment_merge": {
        "enabled": True,
        "max_gap_ms": 2000,
        "max_duration_ms": 20000,
    },
    "vad_options": {},
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json5(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = _strip_json5_comments(text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    if not text.strip():
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return loaded


def merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def split_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip().lower().lstrip(".") for item in value if str(item).strip()]
    return [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _strip_json5_comments(text: str) -> str:
    result: list[str] = []
    index = 0
    in_string = False
    quote = ""
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            result.append(char)
            if char == "\\" and nxt:
                index += 1
                result.append(text[index])
            elif char == quote:
                if quote == "'":
                    result[-1] = '"'
                in_string = False
            index += 1
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            result.append('"' if char == "'" else char)
            index += 1
            continue
        if char == "/" and nxt == "/":
            index = text.find("\n", index)
            if index == -1:
                break
            result.append("\n")
            index += 1
            continue
        if char == "/" and nxt == "*":
            end = text.find("*/", index + 2)
            index = len(text) if end == -1 else end + 2
            continue
        result.append(char)
        index += 1
    return "".join(result)
