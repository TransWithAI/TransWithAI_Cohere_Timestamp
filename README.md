# TransWithAI Cohere Timestamp

This app keeps the reference Faster Whisper UI/CLI shape while delegating transcription, VAD, forced alignment, and subtitle writing to a project-built `crispasr` binary.

The default output is Japanese subtitles: `language: "ja"` and `sub_formats: ["srt"]` in `generation_config.json5`.

## Quick Start

```bash
scripts/setup.sh cpu
python infer.py --help
python infer.py --dry_run --device cpu path/to/audio_or_folder
python infer.py --device cuda --sub_formats srt,vtt,lrc path/to/audio_or_folder
```

`scripts/setup.sh` initializes the CrispASR submodule, compiles the selected backend, downloads the pinned Cohere, Qwen3, and Whisper VAD GGUF models, runs tests, and performs a wrapper dry-run.

Backend choices:

```bash
scripts/setup.sh cpu
scripts/setup.sh cuda128
scripts/setup.sh cuda122
scripts/setup.sh cuda118
scripts/setup.sh vulkan
```

Use `--skip-models` when you only want to compile CrispASR, or `--skip-tests` when rebuilding quickly.

Build scripts cap parallel compilation at 16 jobs by default. You can set `JOBS=1` through `JOBS=16`, but higher values are capped to 16.

The wrapper defaults `crispasr_path` to `auto`, which checks these local build outputs before falling back to `crispasr` on `PATH`:

- `runtimes/<platform>/cuda128/bin/crispasr`
- `runtimes/<platform>/cuda122/bin/crispasr`
- `runtimes/<platform>/cuda118/bin/crispasr`
- `runtimes/<platform>/vulkan/bin/crispasr`
- `runtimes/<platform>/cpu/bin/crispasr`
- `build/crispasr-cuda128/bin/crispasr`
- `build/crispasr-cuda122/bin/crispasr`
- `build/crispasr-cuda118/bin/crispasr`
- `build/crispasr-vulkan/bin/crispasr`
- `build/crispasr-cpu/bin/crispasr`

## Compatibility Flags

Preserved practical flags include `--model_name_or_path`, `--device`, `--compute_type`, `--overwrite`, `--audio_suffixes`, `--sub_formats`, `--output_dir`, `--generation_config`, `--log_level`, VAD override flags, merge segment flags, and positional files/directories.

Configuration precedence is:

1. Built-in defaults.
2. `generation_config.json5` values.
3. CLI flags that are explicitly present.

Faster Whisper-only knobs such as `--compute_type`, fine-grained VAD thresholds, and batching flags are accepted for launcher compatibility but are not reimplemented in Python. CrispASR owns runtime batching, ASR, VAD, alignment, and subtitle writing.

## CrispASR Command Shape

For an input file, this wrapper builds commands like:

```bash
build/crispasr-cpu/bin/crispasr --backend cohere -m models/cohere/model.gguf -f input.mp3 --vad -vm models/vad/whisper-vad-asmr-q4_k.gguf -am models/qwen3-forced-aligner.gguf --force-aligner -l ja --split-on-punct -of input -osrt
```

Subtitle format mapping:

- `srt` -> `-osrt`
- `vtt` -> `-ovtt`
- `lrc` -> `-olrc`
- `txt` -> `-otxt`
- `json` -> `-oj`
- `json-full` or `json_full` -> `-ojf`
- `csv` -> `-ocsv`

Device mapping:

- `--device cpu` -> `--gpu-backend cpu`
- `--device cuda` -> `--gpu-backend cuda`
- `--device amd`, `rocm`, or `hip` -> `--gpu-backend vulkan`
- `--device auto` omits `--gpu-backend`
- `gpu_backend` in config overrides the automatic mapping

`-vm models/vad/whisper-vad-asmr-q4_k.gguf` is explicit because CrispASR defaults to Silero VAD and the `whisper-vad` alias resolves through CrispASR's cache. `-am` is explicit because CrispASR `-am auto` maps to canary CTC, not the requested Qwen3 forced aligner.

## File Discovery And Outputs

Positional files and directories are accepted. Directories are scanned recursively by suffix. Existing output files are skipped unless `--overwrite` is provided. When `--output_dir` is set, relative directory structure is mirrored under that directory; otherwise outputs are written beside the source audio.

Default CrispASR builds support `wav`, `flac`, `mp3`, and `ogg`. Convert container/video formats such as `m4a`, `mp4`, `webm`, `mkv`, or `mov` to one of those audio formats before running this wrapper unless you build CrispASR with additional FFmpeg support yourself.

`latest.log` is overwritten on every run and captures DEBUG-level logs.

## Models

`download_models.py` reads `models/manifest.json5`, downloads artifacts over HTTPS, writes only under `models/`, and requires SHA-256 verification before download. The manifest includes the TransWithAI Japanese Cohere ASR Q6_K model, Qwen3 forced aligner, and Whisper VAD sidecar so release bundles can run without resolving CrispASR's VAD cache alias.

```bash
python download_models.py --dry_run
python download_models.py
```

## Submodule

CrispASR is tracked as `third_party/CrispASR` through `.gitmodules`. See `docs/submodules.md` for checkout requirements.

The build scripts compile the executable target `crispasr-cli`, which outputs a binary named `crispasr`:

```bash
scripts/build-crispasr-cpu.sh
scripts/build-crispasr-cuda118.sh
scripts/build-crispasr-cuda122.sh
scripts/build-crispasr-cuda128.sh
scripts/build-crispasr-vulkan.sh
```

Vulkan builds require a `glslc` shader compiler. In Conda environments this is provided by `shaderc`; on Ubuntu install the `glslc` package alongside `libvulkan-dev` and `spirv-headers`.

CUDA builds are split by runtime line, matching the Faster Whisper release strategy. Each CUDA script is intended to run inside its matching Conda environment with the matching `cuda-compiler` package, then CI removes compiler packages before staging the runtime artifact:

- `environment-cuda118.yml` -> `scripts/build-crispasr-cuda118.sh`
- `environment-cuda122.yml` -> `scripts/build-crispasr-cuda122.sh`
- `environment-cuda128.yml` -> `scripts/build-crispasr-cuda128.sh`

The wrapper selects the newest compatible CUDA runtime based on the installed NVIDIA driver, then falls back to Vulkan or CPU if no CUDA runtime is usable. Set `CUDA_ARCHITECTURES=...` only when you intentionally want a custom architecture list for a build.

## Release Bundles

The release workflow mirrors the reference Faster Whisper workflow style: build runtime artifacts, package archives, upload CI artifacts, and publish them on `v*` tags using large-release upload support.

```bash
python scripts/package-universal.py --runtime cpu --archive dist/transwithai_cohere_timestamp_linux-x64_cpu.zip
python scripts/package-universal.py --runtime cuda128 --archive dist/transwithai_cohere_timestamp_linux-x64_cuda128.zip
python scripts/package-universal.py --archive dist/transwithai_cohere_timestamp_linux-x64_universal.zip
```

Full app bundles always include every artifact listed in `models/manifest.json5`; run `python download_models.py` before packaging. Unlike the reference Faster Whisper release matrix, this project does not publish an empty-model variant.

Per-runtime bundles are complete downloadable app bundles with the launch `.bat` files, project source/docs, `models/`, and exactly one runtime under `runtimes/linux-x64/<runtime>/`. Release builds publish these runtime-specific archives:

- `transwithai_cohere_timestamp_linux-x64_cpu.zip`
- `transwithai_cohere_timestamp_linux-x64_vulkan.zip`
- `transwithai_cohere_timestamp_linux-x64_cuda118.zip`
- `transwithai_cohere_timestamp_linux-x64_cuda122.zip`
- `transwithai_cohere_timestamp_linux-x64_cuda128.zip`

The universal bundle is also published as `transwithai_cohere_timestamp_linux-x64_universal.zip` and includes all staged runtimes:

```text
runtimes/linux-x64/cpu/bin/crispasr
runtimes/linux-x64/cuda118/bin/crispasr
runtimes/linux-x64/cuda122/bin/crispasr
runtimes/linux-x64/cuda128/bin/crispasr
runtimes/linux-x64/vulkan/bin/crispasr
```

Each CUDA runtime is built in a separate Conda environment with the appropriate CUDA compiler/runtime line. Pick one per-runtime archive for a smaller direct download, or use the universal archive when you want one bundle that can choose among all available runtimes automatically.

## CUDA/Vulkan Notes

This wrapper does not depend on PyTorch, qwen-asr, transformers, or the Cohere SDK. CrispASR is a native binary project, so its support matrix differs from Faster Whisper. See `docs/cuda-support.md` and the environment sketches for CPU, CUDA 12.8, and Vulkan-first builds.
