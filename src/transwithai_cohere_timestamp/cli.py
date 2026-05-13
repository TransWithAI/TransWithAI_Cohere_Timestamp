from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .config import BUILTIN_DEFAULTS, coerce_bool, load_json5, merge_config, project_root, split_csv
from .crispasr import FORMAT_FLAGS, build_command, command_to_string, run_task
from .discovery import discover_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TransWithAI CrispASR Cohere subtitle wrapper")
    parser.add_argument("base_dirs", nargs="*", help="Audio files or directories to process recursively.")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "amd", "rocm", "hip"])
    parser.add_argument("--compute_type")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audio_suffixes")
    parser.add_argument("--sub_formats")
    parser.add_argument("--output_dir")
    parser.add_argument("--generation_config", default="generation_config.json5")
    parser.add_argument("--log_level")
    parser.add_argument("--language")
    parser.add_argument("--crispasr_path")
    parser.add_argument("--qwen3_forced_aligner_path")
    parser.add_argument("--gpu_backend")
    parser.add_argument("--dry_run", action="store_true", help="Print CrispASR commands without executing them.")
    parser.add_argument("--console", action="store_true", help="Compatibility flag; logging is always enabled.")
    parser.add_argument("--vad_threshold", type=float)
    parser.add_argument("--vad_min_speech_duration_ms", type=int)
    parser.add_argument("--vad_min_silence_duration_ms", type=int)
    parser.add_argument("--vad_speech_pad_ms", type=int)
    parser.add_argument("--vad_filter")
    parser.add_argument("--vad_model")
    parser.add_argument("--merge_segments", action="store_true")
    parser.add_argument("--no_merge_segments", action="store_true")
    parser.add_argument("--merge_max_gap_ms", type=int)
    parser.add_argument("--merge_max_duration_ms", type=int)
    parser.add_argument("--enable_batching", action="store_true", help="Compatibility no-op; CrispASR owns batching.")
    parser.add_argument("--batch_size", type=int, help="Compatibility no-op; CrispASR owns batching.")
    parser.add_argument("--max_batch_size", type=int, help="Compatibility no-op; CrispASR owns batching.")
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.generation_config)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    config = merge_config(BUILTIN_DEFAULTS, load_json5(config_path))
    vad_parameters = config.pop("vad_parameters", None)
    if isinstance(vad_parameters, dict):
        config["vad_options"] = merge_config(dict(config.get("vad_options") or {}), vad_parameters)
    cli_values = {
        "model_name_or_path": args.model_name_or_path,
        "device": args.device,
        "compute_type": args.compute_type,
        "audio_suffixes": split_csv(args.audio_suffixes),
        "sub_formats": split_csv(args.sub_formats),
        "output_dir": args.output_dir,
        "log_level": args.log_level,
        "language": args.language,
        "crispasr_path": args.crispasr_path,
        "qwen3_forced_aligner_path": args.qwen3_forced_aligner_path,
        "gpu_backend": args.gpu_backend,
        "vad_model": args.vad_model,
    }
    for key, value in cli_values.items():
        if value is not None:
            config[key] = value
    if args.overwrite:
        config["overwrite"] = True
    if args.dry_run:
        config["dry_run"] = True
    if args.vad_filter is not None:
        config["vad_filter"] = coerce_bool(args.vad_filter)
    vad_options = dict(config.get("vad_options") or {})
    for key in ("vad_threshold", "vad_min_speech_duration_ms", "vad_min_silence_duration_ms", "vad_speech_pad_ms"):
        value = getattr(args, key)
        if value is not None:
            vad_options[key] = value
    config["vad_options"] = vad_options
    segment_merge = dict(config.get("segment_merge") or {})
    if args.merge_segments:
        segment_merge["enabled"] = True
    if args.no_merge_segments:
        segment_merge["enabled"] = False
    if args.merge_max_gap_ms is not None:
        segment_merge["max_gap_ms"] = args.merge_max_gap_ms
    if args.merge_max_duration_ms is not None:
        segment_merge["max_duration_ms"] = args.merge_max_duration_ms
    config["segment_merge"] = segment_merge
    config["audio_suffixes"] = split_csv(config.get("audio_suffixes")) or []
    config["sub_formats"] = split_csv(config.get("sub_formats")) or []
    _resolve_runtime_paths(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    unsupported = [fmt for fmt in config.get("sub_formats", []) if fmt not in FORMAT_FLAGS]
    if unsupported:
        raise ValueError(f"Unsupported subtitle format(s): {', '.join(unsupported)}")


def setup_logging(level_name: str) -> logging.Logger:
    logger = logging.getLogger("transwithai_cohere_timestamp")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(project_root() / "latest.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level_name.upper(), logging.DEBUG))
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _resolve_runtime_paths(config: dict[str, Any]) -> None:
    root = project_root()
    for key in ("model_name_or_path", "qwen3_forced_aligner_path", "vad_model"):
        value = config.get(key)
        if isinstance(value, str) and value and _should_resolve_model_path(key, value):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = root / path
            config[key] = str(path)
    crispasr_path = str(config.get("crispasr_path") or "auto")
    if crispasr_path.lower() in {"auto", "local", ""}:
        config["crispasr_path"] = str(_discover_local_crispasr(config, root) or "crispasr")
    else:
        expanded = Path(crispasr_path).expanduser()
        if expanded.parts and (expanded.is_absolute() or len(expanded.parts) > 1):
            if not expanded.is_absolute():
                expanded = root / expanded
            config["crispasr_path"] = str(expanded)


def _is_remote_or_auto(value: str) -> bool:
    lower = value.lower()
    return lower == "auto" or lower.startswith(("http://", "https://", "hf://"))


def _should_resolve_model_path(key: str, value: str) -> bool:
    if _is_remote_or_auto(value):
        return False
    if key == "vad_model" and _is_named_vad_alias(value):
        return False
    return True


def _is_named_vad_alias(value: str) -> bool:
    lower = value.lower()
    if lower.endswith((".gguf", ".bin", ".onnx")):
        return False
    return "/" not in value and "\\" not in value


def _discover_local_crispasr(config: dict[str, Any], root: Path) -> Path | None:
    executable = "crispasr.exe" if os.name == "nt" else "crispasr"
    runtime_root = root / "runtimes" / _runtime_platform_tag()
    cuda_paths = []
    for runtime_name in _cuda_runtime_order():
        cuda_paths.extend(
            [
                runtime_root / runtime_name / "bin" / executable,
                root / "build" / f"crispasr-{runtime_name}" / "bin" / executable,
            ]
        )
    by_backend = {
        "cuda": cuda_paths,
        "vulkan": [
            runtime_root / "vulkan" / "bin" / executable,
            root / "build" / "crispasr-vulkan" / "bin" / executable,
        ],
        "cpu": [
            runtime_root / "cpu" / "bin" / executable,
            root / "build" / "crispasr-cpu" / "bin" / executable,
        ],
    }
    requested = _requested_build_backend(config)
    candidates: list[Path] = []
    if requested in by_backend:
        for candidate in by_backend[requested]:
            if candidate.is_file():
                return candidate
        return None
    for backend in ("cuda", "vulkan", "cpu"):
        candidates.extend(by_backend[backend])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _cuda_runtime_order() -> list[str]:
    driver = _nvidia_driver_version()
    runtimes = [
        ("cuda128", (570, 0)),
        ("cuda122", (535, 54)),
        ("cuda118", (520, 61)),
    ]
    if driver is None:
        return [name for name, _minimum in runtimes]
    return [name for name, minimum in runtimes if driver >= minimum]


def _nvidia_driver_version() -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parsed = _parse_driver_version(line.strip())
        if parsed is not None:
            return parsed
    return None


def _parse_driver_version(value: str) -> tuple[int, int] | None:
    parts = value.split(".")
    if not parts or not parts[0].isdigit():
        return None
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def _runtime_platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        arch = "x64"
    elif machine in {"arm64", "aarch64"}:
        arch = "arm64"
    else:
        arch = machine or "unknown"
    if system == "windows":
        os_tag = "windows"
    elif system == "darwin":
        os_tag = "macos"
    else:
        os_tag = "linux"
    return f"{os_tag}-{arch}"


def _requested_build_backend(config: dict[str, Any]) -> str | None:
    backend = str(config.get("gpu_backend") or "").lower()
    if backend in {"cuda", "vulkan", "cpu"}:
        return backend
    device = str(config.get("device") or "auto").lower()
    if device == "cuda":
        return "cuda"
    if device in {"amd", "rocm", "hip"}:
        return "vulkan"
    if device == "cpu":
        return "cpu"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
        validate_config(config)
    except Exception as exc:
        parser.error(str(exc))
    logger = setup_logging(str(config.get("log_level") or "DEBUG"))
    _log_compatibility_notes(config, logger)
    if not args.base_dirs:
        parser.print_help()
        return 0
    tasks = discover_audio(
        args.base_dirs,
        list(config["audio_suffixes"]),
        config.get("output_dir"),
        list(config["sub_formats"]),
        bool(config.get("overwrite")),
    )
    if not tasks:
        logger.info("No inputs need processing. Check paths, suffixes, or existing outputs.")
        return 0
    failures = 0
    for task in tasks:
        command = build_command(task, config)
        if config.get("dry_run"):
            print(command_to_string(command))
        result = run_task(task, config, logger)
        if result.returncode != 0:
            failures += 1
            logger.error("CrispASR failed for %s with exit code %s", task.source, result.returncode)
    return 1 if failures else 0


def _log_compatibility_notes(config: dict[str, Any], logger: logging.Logger) -> None:
    if config.get("compute_type") not in (None, "auto", "default"):
        logger.warning("--compute_type is accepted for Faster Whisper compatibility but is not passed to CrispASR.")
    unsupported_vad_options = set(config.get("vad_options") or {}) - {
        "vad_threshold",
        "threshold",
        "vad_min_speech_duration_ms",
        "min_speech_duration_ms",
        "vad_min_silence_duration_ms",
        "min_silence_duration_ms",
        "vad_speech_pad_ms",
        "speech_pad_ms",
    }
    if unsupported_vad_options:
        logger.warning("Some VAD overrides are accepted for compatibility but not passed to CrispASR: %s", ", ".join(sorted(unsupported_vad_options)))
    if not config.get("segment_merge", {}).get("enabled", True):
        logger.warning("Segment merge flags are accepted for compatibility; CrispASR subtitle segmentation is delegated unchanged.")
if __name__ == "__main__":
    raise SystemExit(main())
