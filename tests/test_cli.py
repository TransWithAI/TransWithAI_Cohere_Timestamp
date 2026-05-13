from __future__ import annotations
# pyright: reportMissingImports=false

from pathlib import Path
from unittest.mock import Mock

import transwithai_cohere_timestamp.cli as cli
from transwithai_cohere_timestamp.cli import build_parser, resolve_config, validate_config
from transwithai_cohere_timestamp.crispasr import build_command, run_task
from transwithai_cohere_timestamp.discovery import AudioTask, discover_audio


def test_config_precedence_defaults_to_japanese_srt() -> None:
    args = build_parser().parse_args(["--generation_config", "missing.json5"])
    config = resolve_config(args)
    assert config["language"] == "ja"
    assert config["sub_formats"] == ["srt"]


def test_command_builds_explicit_vad_aligner_and_vulkan_for_amd(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"")
    task = AudioTask(source=audio, output_base=tmp_path / "sample", missing_formats=("srt", "vtt"))
    config = resolve_config(build_parser().parse_args(["--generation_config", "missing.json5", "--device", "amd"]))
    command = build_command(task, config)
    assert Path(command[0]).name == "crispasr"
    assert command[1:4] == ["--backend", "cohere", "-m"]
    assert Path(command[4]).name == "model.gguf"
    assert Path(command[4]).is_absolute()
    assert "--vad" in command
    vad_path = Path(command[command.index("-vm") + 1])
    assert vad_path.name == "whisper-vad-asmr-q4_k.gguf"
    assert vad_path.is_absolute()
    aligner_path = Path(command[command.index("-am") + 1])
    assert aligner_path.name == "qwen3-forced-aligner.gguf"
    assert aligner_path.is_absolute()
    assert "--force-aligner" in command
    assert command[command.index("--gpu-backend") + 1] == "vulkan"
    assert "-osrt" in command
    assert "-ovtt" in command


def test_auto_crispasr_path_discovers_matching_local_build(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "build" / "crispasr-cpu" / "bin" / "crispasr"
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    config = cli.resolve_config(cli.build_parser().parse_args(["--generation_config", "missing.json5", "--device", "cpu"]))
    assert config["crispasr_path"] == str(binary)
    assert config["model_name_or_path"] == str(tmp_path / "models" / "cohere" / "model.gguf")
    assert config["vad_model"] == str(tmp_path / "models" / "vad" / "whisper-vad-asmr-q4_k.gguf")


def test_vad_model_alias_is_preserved(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json5"
    config_path.write_text('{"vad_model": "whisper-vad"}', encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    config = cli.resolve_config(cli.build_parser().parse_args(["--generation_config", str(config_path)]))
    assert config["vad_model"] == "whisper-vad"


def test_auto_crispasr_path_prefers_packaged_runtime(tmp_path: Path, monkeypatch) -> None:
    runtime_binary = tmp_path / "runtimes" / "linux-x64" / "cuda128" / "bin" / "crispasr"
    build_binary = tmp_path / "build" / "crispasr-cuda128" / "bin" / "crispasr"
    runtime_binary.parent.mkdir(parents=True)
    build_binary.parent.mkdir(parents=True)
    runtime_binary.write_text("", encoding="utf-8")
    build_binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_runtime_platform_tag", lambda: "linux-x64")
    config = cli.resolve_config(cli.build_parser().parse_args(["--generation_config", "missing.json5", "--device", "cuda"]))
    assert config["crispasr_path"] == str(runtime_binary)


def test_cuda_runtime_selection_uses_driver_compatible_runtime(tmp_path: Path, monkeypatch) -> None:
    cuda128 = tmp_path / "runtimes" / "linux-x64" / "cuda128" / "bin" / "crispasr"
    cuda122 = tmp_path / "runtimes" / "linux-x64" / "cuda122" / "bin" / "crispasr"
    cuda118 = tmp_path / "runtimes" / "linux-x64" / "cuda118" / "bin" / "crispasr"
    for binary in (cuda128, cuda122, cuda118):
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_runtime_platform_tag", lambda: "linux-x64")
    monkeypatch.setattr(cli, "_nvidia_driver_version", lambda: (535, 80))
    config = cli.resolve_config(cli.build_parser().parse_args(["--generation_config", "missing.json5", "--device", "cuda"]))
    assert config["crispasr_path"] == str(cuda122)


def test_explicit_cuda_does_not_fall_back_to_cpu_binary(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "build" / "crispasr-cpu" / "bin" / "crispasr"
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    config = cli.resolve_config(cli.build_parser().parse_args(["--generation_config", "missing.json5", "--device", "cuda"]))
    assert config["crispasr_path"] == "crispasr"


def test_discovery_skips_existing_outputs_unless_overwrite(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    audio.with_suffix(".srt").write_text("existing", encoding="utf-8")
    assert discover_audio([str(audio)], ["wav"], None, ["srt"], False) == []
    tasks = discover_audio([str(audio)], ["wav"], None, ["srt"], True)
    assert len(tasks) == 1
    assert tasks[0].output_base == audio.with_suffix("")


def test_json_full_checks_json_output_extension(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    audio.with_suffix(".json").write_text("{}", encoding="utf-8")
    assert discover_audio([str(audio)], ["wav"], None, ["json-full"], False) == []


def test_vad_speech_pad_maps_to_crispasr_flag(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"")
    task = AudioTask(source=audio, output_base=tmp_path / "sample", missing_formats=("srt",))
    config = resolve_config(
        build_parser().parse_args(["--generation_config", "missing.json5", "--vad_speech_pad_ms", "120"])
    )
    command = build_command(task, config)
    assert command[command.index("-vp") + 1] == "120"


def test_invalid_subtitle_format_validates_cleanly() -> None:
    config = resolve_config(build_parser().parse_args(["--generation_config", "missing.json5", "--sub_formats", "ass"]))
    try:
        validate_config(config)
    except ValueError as exc:
        assert "Unsupported subtitle format" in str(exc)
    else:
        raise AssertionError("expected unsupported format validation error")


def test_single_quoted_config_string_is_supported(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json5"
    config_path.write_text("{'language': 'ko', 'sub_formats': ['vtt']}", encoding="utf-8")
    config = resolve_config(build_parser().parse_args(["--generation_config", str(config_path)]))
    assert config["language"] == "ko"
    assert config["sub_formats"] == ["vtt"]


def test_dry_run_does_not_create_output_directory(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"")
    output_base = tmp_path / "new-output" / "sample"
    task = AudioTask(source=audio, output_base=output_base, missing_formats=("srt",))
    config = resolve_config(build_parser().parse_args(["--generation_config", "missing.json5", "--dry_run"]))
    result = run_task(task, config, Mock())
    assert result.returncode == 0
    assert not output_base.parent.exists()
