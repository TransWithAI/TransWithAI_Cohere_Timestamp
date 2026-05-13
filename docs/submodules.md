# Submodule Requirements

This repository expects CrispASR at `third_party/CrispASR`:

```bash
git submodule update --init --recursive
```

The parent repository should keep `.gitmodules` with:

```ini
[submodule "third_party/CrispASR"]
	path = third_party/CrispASR
	url = https://github.com/CrispStrobe/CrispASR.git
```

Do not edit files under `third_party/CrispASR` from this wrapper project. Build CrispASR following upstream instructions, then set `crispasr_path` in `generation_config.json5` or place the resulting binary on `PATH`.
