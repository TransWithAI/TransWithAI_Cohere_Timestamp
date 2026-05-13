#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
APP_NAME = "transwithai_cohere_timestamp"
ELF_SYSTEM_LIBS = {
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "linux-vdso.so.1",
}
HOST_CUDA_DRIVER_LIB_PREFIXES = ("libcuda.so", "libnvidia-")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a universal TransWithAI Cohere Timestamp bundle.")
    parser.add_argument("--output", default=str(DIST / APP_NAME), help="Output directory before zipping.")
    parser.add_argument("--archive", default=None, help="Optional zip path to create.")
    parser.add_argument("--include-models", action="store_true", help="Deprecated no-op; full app bundles always include models.")
    parser.add_argument("--platform-tag", default=platform_tag(), help="Runtime platform tag, e.g. linux-x64 or windows-x64.")
    parser.add_argument("--runtime", action="append", dest="runtimes", help="Runtime to include in a full app bundle. May be repeated.")
    parser.add_argument("--stage-runtime", action="store_true", help="Stage one runtime directory instead of a full app bundle.")
    parser.add_argument("--runtime-name", default=None, help="Runtime name for --stage-runtime, e.g. cuda118.")
    parser.add_argument("--runtime-build-dir", default=None, help="Build directory for --stage-runtime.")
    parser.add_argument("--runtime-output", default=None, help="Output directory for --stage-runtime.")
    parser.add_argument("--check-runtime-linkage", action="store_true", help="Run ldd on one runtime binary and fail on unresolved bundled dependencies.")
    parser.add_argument("--runtime-binary", default=None, help="Runtime executable for --check-runtime-linkage.")
    parser.add_argument("--allow-missing-cuda-driver", action="store_true", help="Allow unresolved libcuda/libnvidia driver libraries during linkage checks.")
    args = parser.parse_args()

    if args.check_runtime_linkage:
        if not args.runtime_binary:
            parser.error("--check-runtime-linkage requires --runtime-binary")
        check_runtime_linkage(Path(args.runtime_binary), args.allow_missing_cuda_driver)
        return 0

    if args.stage_runtime:
        if not args.runtime_name or not args.runtime_build_dir or not args.runtime_output:
            parser.error("--stage-runtime requires --runtime-name, --runtime-build-dir, and --runtime-output")
        output = Path(args.runtime_output)
        if output.exists():
            shutil.rmtree(output)
        copy_runtime_tree(Path(args.runtime_build_dir), output, include_cuda_libs=args.runtime_name.startswith("cuda"))
        print(output)
        return 0

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copy_project_files(out)
    copy_runtimes(out, args.platform_tag, set(args.runtimes or []))
    copy_models(out)

    if args.archive:
        make_zip(out, Path(args.archive))
    print(out)
    return 0


def platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "x64" if machine in {"amd64", "x86_64"} else machine or "unknown"
    if system == "windows":
        return f"windows-{arch}"
    if system == "darwin":
        return f"macos-{arch}"
    return f"linux-{arch}"


def copy_project_files(out: Path) -> None:
    files = [
        "infer.py",
        "download_models.py",
        "generation_config.json5",
        "README.md",
        "pyproject.toml",
        "运行(CPU).bat",
        "运行(GPU).bat",
        "运行(GPU,低显存模式).bat",
        "运行(GPU,高显存加速模式).bat",
        "运行(GPU)(输出到当前文件夹).bat",
    ]
    for relative in files:
        source = ROOT / relative
        if source.exists():
            shutil.copy2(source, out / relative)
    for relative in ["src", "docs"]:
        source = ROOT / relative
        if source.exists():
            shutil.copytree(source, out / relative, ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"))


def copy_models(out: Path) -> None:
    model_out = out / "models"
    model_out.mkdir(exist_ok=True)
    manifest = ROOT / "models" / "manifest.json5"
    if not manifest.exists():
        raise FileNotFoundError(f"missing model manifest: {manifest}")
    shutil.copy2(manifest, model_out / "manifest.json5")
    require_manifest_artifacts(manifest)
    for source in (ROOT / "models").glob("**/*"):
        if source.is_file() and source.name != "manifest.json5":
            destination = model_out / source.relative_to(ROOT / "models")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def require_manifest_artifacts(manifest: Path) -> None:
    loaded = json.loads(strip_json5_comments(manifest.read_text(encoding="utf-8")))
    artifacts = loaded.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError(f"manifest artifacts must be a list: {manifest}")
    missing: list[Path] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        relative = artifact.get("path")
        if not isinstance(relative, str):
            continue
        target = ROOT / relative
        if not target.is_file():
            missing.append(target)
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing model artifact(s); run download_models.py first:\n{formatted}")


def strip_json5_comments(text: str) -> str:
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
    return re.sub(r",\s*([}\]])", r"\1", "".join(result))


def copy_runtimes(out: Path, runtime_platform: str, selected_runtimes: set[str] | None = None) -> None:
    runtime_out = out / "runtimes" / runtime_platform
    runtime_out.mkdir(parents=True, exist_ok=True)
    runtime_specs = {
        "cpu": ROOT / "build" / "crispasr-cpu",
        "vulkan": ROOT / "build" / "crispasr-vulkan",
        "cuda118": ROOT / "build" / "crispasr-cuda118",
        "cuda122": ROOT / "build" / "crispasr-cuda122",
        "cuda128": ROOT / "build" / "crispasr-cuda128",
    }
    unknown_runtimes = selected_runtimes.difference(runtime_specs) if selected_runtimes else set()
    if unknown_runtimes:
        raise ValueError(f"unknown runtime(s): {', '.join(sorted(unknown_runtimes))}")
    copied_runtimes: set[str] = set()
    for name, build_dir in runtime_specs.items():
        if selected_runtimes and name not in selected_runtimes:
            continue
        staged = ROOT / "runtimes" / runtime_platform / name
        destination = runtime_out / name
        if staged.exists():
            shutil.copytree(staged, destination, symlinks=True)
            ensure_runtime_executable(destination)
            copied_runtimes.add(name)
            continue
        if not build_dir.exists():
            continue
        copy_runtime_tree(build_dir, destination, include_cuda_libs=name.startswith("cuda"))
        ensure_runtime_executable(destination)
        copied_runtimes.add(name)
    missing_selected = selected_runtimes.difference(copied_runtimes) if selected_runtimes else set()
    if missing_selected:
        raise FileNotFoundError(f"missing selected runtime(s): {', '.join(sorted(missing_selected))}")


def copy_runtime_tree(build_dir: Path, destination: Path, include_cuda_libs: bool) -> None:
    bin_dir = destination / "bin"
    lib_dir = destination / "lib"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    executable = build_dir / "bin" / ("crispasr.exe" if os.name == "nt" else "crispasr")
    if executable.exists():
        shutil.copy2(executable, bin_dir / executable.name)

    patterns = ["*.so", "*.so.*", "*.dll", "*.dylib"]
    for pattern in patterns:
        for source in build_dir.glob(f"**/{pattern}"):
            if source.is_file():
                copy_library(source, lib_dir)
    if include_cuda_libs:
        copy_cuda_runtime_libs(lib_dir)
    stage_dependency_closure(bin_dir / executable.name, lib_dir, allow_missing_cuda_driver=include_cuda_libs)
    patch_runtime_rpaths(bin_dir / executable.name, lib_dir)


def ensure_runtime_executable(runtime_dir: Path) -> None:
    for name in ("crispasr", "crispasr.exe"):
        executable = runtime_dir / "bin" / name
        if executable.exists():
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_cuda_runtime_libs(lib_dir: Path) -> None:
    search_roots = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        search_roots.append(Path(conda_prefix))
    else:
        for env_name in ("CUDA_PATH", "CUDA_HOME"):
            value = os.environ.get(env_name)
            if value:
                search_roots.append(Path(value))
        search_roots.extend([Path("/usr/local/cuda"), Path("/opt/cuda")])
    patterns = [
        "libcudart.so*", "libcublas.so*", "libcublasLt.so*", "libnvrtc.so*", "libnvrtc-builtins.so*",
        "cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "nvrtc64_*.dll", "nvrtc-builtins64_*.dll",
    ]
    for root in search_roots:
        for subdir in (root, root / "lib", root / "lib64", root / "Library" / "bin", root / "bin"):
            if not subdir.exists():
                continue
            for pattern in patterns:
                for source in subdir.glob(pattern):
                    if source.is_file():
                        copy_library(source, lib_dir)


def copy_library(source: Path, lib_dir: Path) -> Path:
    lib_dir.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        resolved = source.resolve()
        if resolved.exists():
            real_dest = copy_library(resolved, lib_dir)
            return ensure_alias(lib_dir, source.name, real_dest.name)
    destination = lib_dir / source.name
    if source.resolve() != destination.resolve() if destination.exists() else True:
        shutil.copy2(source, destination)
    soname = read_soname(destination)
    if soname and soname != destination.name:
        ensure_alias(lib_dir, soname, destination.name)
    return destination


def ensure_alias(lib_dir: Path, alias: str, target_name: str) -> Path:
    destination = lib_dir / alias
    if alias == target_name:
        return destination
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(target_name)
    except OSError:
        shutil.copy2(lib_dir / target_name, destination)
    return destination


def read_soname(path: Path) -> str | None:
    readelf = shutil.which("readelf")
    if not readelf or path.suffix not in {".so", ""} and ".so" not in path.name:
        return None
    result = subprocess.run([readelf, "-d", str(path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    match = re.search(r"\(SONAME\).*\[(?P<soname>[^\]]+)\]", result.stdout)
    return match.group("soname") if match else None


def stage_dependency_closure(executable: Path, lib_dir: Path, allow_missing_cuda_driver: bool = False) -> None:
    if platform.system().lower() != "linux" or not shutil.which("ldd"):
        return
    pending = [executable, *sorted(lib_dir.glob("*.so*"))]
    seen_inputs: set[Path] = set()
    copied_names = {path.name for path in lib_dir.glob("*")}
    while pending:
        current = pending.pop()
        resolved_current = current.resolve()
        if resolved_current in seen_inputs or not current.exists():
            continue
        seen_inputs.add(resolved_current)
        for dependency in ldd_dependencies(current, allow_missing_cuda_driver=allow_missing_cuda_driver):
            if should_skip_dependency(dependency):
                continue
            if dependency.name in copied_names:
                continue
            copied = copy_library(dependency, lib_dir)
            copied_names.add(copied.name)
            pending.append(copied)


def ldd_dependencies(path: Path, allow_missing_cuda_driver: bool = False) -> list[Path]:
    stdout = run_ldd(path)
    dependencies: list[Path] = []
    for line in stdout.splitlines():
        parsed = parse_ldd_line(line, allow_missing_cuda_driver=allow_missing_cuda_driver)
        if parsed is not None:
            dependencies.append(parsed)
    return dependencies


def check_runtime_linkage(executable: Path, allow_missing_cuda_driver: bool) -> None:
    if platform.system().lower() != "linux" or not shutil.which("ldd"):
        return
    if not executable.is_file():
        raise FileNotFoundError(f"runtime executable not found: {executable}")
    stdout = run_ldd(executable)
    print(stdout, end="" if stdout.endswith("\n") else "\n")
    for line in stdout.splitlines():
        parse_ldd_line(line, allow_missing_cuda_driver=allow_missing_cuda_driver)


def run_ldd(path: Path) -> str:
    result = subprocess.run(["ldd", str(path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ldd failed for {path}:\n{result.stderr or result.stdout}")
    return result.stdout


def parse_ldd_line(line: str, allow_missing_cuda_driver: bool = False) -> Path | None:
    missing_dependency = parse_missing_ldd_dependency(line)
    if missing_dependency:
        if allow_missing_cuda_driver and is_host_cuda_driver_lib(missing_dependency):
            return None
        raise RuntimeError(f"unresolved dependency while packaging: {line.strip()}")
    match = re.search(r"=>\s+(?P<path>/\S+)", line)
    if not match:
        match = re.match(r"\s*(?P<path>/\S+)", line)
    if not match:
        return None
    path = Path(match.group("path"))
    return path if path.exists() else None


def should_skip_dependency(path: Path) -> bool:
    name = path.name
    if name in ELF_SYSTEM_LIBS:
        return True
    if is_host_cuda_driver_lib(name):
        return True
    return False


def parse_missing_ldd_dependency(line: str) -> str | None:
    match = re.match(r"\s*(?P<name>\S+)\s+=>\s+not found", line)
    return match.group("name") if match else None


def is_host_cuda_driver_lib(name: str) -> bool:
    return name.startswith(HOST_CUDA_DRIVER_LIB_PREFIXES)


def patch_runtime_rpaths(executable: Path, lib_dir: Path) -> None:
    if platform.system().lower() != "linux":
        return
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to package Linux runtimes with bundle-relative RPATH")
    patch_rpath(executable, "$ORIGIN/../lib")
    for library in sorted(lib_dir.glob("*.so*")):
        if library.is_file() and not library.is_symlink():
            patch_rpath(library, "$ORIGIN")


def patch_rpath(path: Path, rpath: str) -> None:
    subprocess.run(["patchelf", "--set-rpath", rpath, str(path)], check=True)


def make_zip(source_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        for path in source_dir.rglob("*"):
            arcname = path.relative_to(source_dir.parent)
            if path.is_symlink():
                info = zipfile.ZipInfo(str(arcname))
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zf.writestr(info, os.readlink(path))
            elif path.is_file():
                zf.write(path, arcname)


if __name__ == "__main__":
    raise SystemExit(main())
