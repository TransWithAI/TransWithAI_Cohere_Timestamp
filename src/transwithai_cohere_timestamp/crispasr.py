from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .discovery import AudioTask


FORMAT_FLAGS = {
    "srt": "-osrt",
    "vtt": "-ovtt",
    "lrc": "-olrc",
    "txt": "-otxt",
    "json": "-oj",
    "json-full": "-ojf",
    "json_full": "-ojf",
    "csv": "-ocsv",
}

DEVICE_TO_GPU_BACKEND = {
    "cpu": "cpu",
    "cuda": "cuda",
    "amd": "vulkan",
    "rocm": "vulkan",
    "hip": "vulkan",
}


@dataclass(frozen=True)
class RunResult:
    task: AudioTask
    command: list[str]
    returncode: int


def build_command(task: AudioTask, config: dict[str, Any], formats: list[str] | None = None) -> list[str]:
    selected_formats = formats or list(task.missing_formats)
    command = [
        str(config["crispasr_path"]),
        "--backend",
        "cohere",
        "-m",
        str(config["model_name_or_path"]),
        "-f",
        str(task.source),
    ]
    if bool(config.get("vad_filter", True)):
        command.append("--vad")
        command.extend(["-vm", str(config.get("vad_model") or "whisper-vad")])
        command.extend(_vad_option_flags(config.get("vad_options") or {}))
    command.extend(["-am", str(config["qwen3_forced_aligner_path"])])
    if bool(config.get("force_aligner", True)):
        command.append("--force-aligner")
    command.extend(["-l", str(config.get("language") or "ja")])
    if bool(config.get("split_on_punct", True)):
        command.append("--split-on-punct")
    gpu_backend = _resolve_gpu_backend(config)
    if gpu_backend:
        command.extend(["--gpu-backend", gpu_backend])
    command.extend(["-of", str(task.output_base)])
    for fmt in selected_formats:
        normalized = fmt.lower().strip().lstrip(".")
        try:
            command.append(FORMAT_FLAGS[normalized])
        except KeyError as exc:
            raise ValueError(f"Unsupported subtitle format for CrispASR: {fmt}") from exc
    return command


def run_task(task: AudioTask, config: dict[str, Any], logger: logging.Logger) -> RunResult:
    command = build_command(task, config)
    logger.info("Running: %s", command_to_string(command))
    if config.get("dry_run"):
        return RunResult(task=task, command=command, returncode=0)
    task.output_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        logger.error("CrispASR binary not found: %s", command[0])
        logger.error("Run scripts/setup.sh, put crispasr on PATH, or set crispasr_path in generation_config.json5.")
        return RunResult(task=task, command=command, returncode=127)
    except OSError as exc:
        logger.error("Failed to start CrispASR: %s", exc)
        return RunResult(task=task, command=command, returncode=126)
    return RunResult(task=task, command=command, returncode=completed.returncode)


def _resolve_gpu_backend(config: dict[str, Any]) -> str | None:
    configured = config.get("gpu_backend")
    if configured:
        return str(configured)
    device = str(config.get("device") or "auto").lower()
    if device == "auto":
        return None
    return DEVICE_TO_GPU_BACKEND.get(device, device)


def _vad_option_flags(vad_options: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    mappings = {
        "vad_threshold": "-vt",
        "threshold": "-vt",
        "vad_min_speech_duration_ms": "-vspd",
        "min_speech_duration_ms": "-vspd",
        "vad_min_silence_duration_ms": "-vsd",
        "min_silence_duration_ms": "-vsd",
        "vad_speech_pad_ms": "-vp",
        "speech_pad_ms": "-vp",
    }
    for key, flag in mappings.items():
        value = vad_options.get(key)
        if value is not None:
            flags.extend([flag, str(value)])
    return flags


def command_to_string(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)
