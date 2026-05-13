# CUDA, CPU, And Vulkan Support Notes

The reference Faster Whisper project used a Python/PyTorch-adjacent packaging matrix. This wrapper does not. Runtime inference is delegated to the native `crispasr` binary, so Python only needs the standard library plus optional test tooling.

Recommended investigation order:

1. CPU build: baseline correctness and portability.
2. CUDA 12.8 build: first NVIDIA target to validate.
3. Vulkan build: default path for AMD/ROCm/HIP aliases because CrispASR is not a PyTorch ROCm runtime.

CUDA 11.8, CUDA 12.2, and CUDA 12.8 are built as separate runtime artifacts in CI, using separate Conda environments with matching `cuda-compiler`, `cuda-libraries-dev`, and `cuda-runtime` packages.

CI removes compiler packages after compilation and before staging the runtime artifact. The packaged app carries runtime libraries, not the full CUDA compiler/toolkit. `libcuda` still comes from the user's NVIDIA driver, so CPU-only CI validates CUDA linkage while allowing that host-provided driver library to be unresolved.

The runtime selector chooses the newest packaged CUDA runtime compatible with the detected NVIDIA driver:

- CUDA 12.8 runtime for drivers roughly `>= 570`.
- CUDA 12.2 runtime for drivers roughly `>= 535.54`.
- CUDA 11.8 runtime for drivers roughly `>= 520.61`.

If no compatible CUDA runtime is found, users should use Vulkan or CPU.

Wrapper device mapping:

- `cpu` maps to `--gpu-backend cpu`.
- `cuda` maps to `--gpu-backend cuda`.
- `amd`, `rocm`, and `hip` map to `--gpu-backend vulkan` unless `gpu_backend` is set in config.
- `auto` omits `--gpu-backend` and leaves selection to CrispASR.
