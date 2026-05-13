from __future__ import annotations
# pyright: reportMissingImports=false

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transwithai_cohere_timestamp.config import load_json5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model artifacts listed in models/manifest.json5.")
    parser.add_argument("--manifest", default="models/manifest.json5")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = load_json5(manifest_path)
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise SystemExit("manifest artifacts must be a list")
    for artifact in artifacts:
        target = resolve_artifact_path(str(artifact["path"]))
        url = artifact.get("url")
        expected_sha256 = artifact.get("sha256")
        if not url or "TODO" in url:
            print(f"SKIP {target}: URL is a placeholder; update manifest first.")
            continue
        validate_url(str(url))
        validate_sha256(str(expected_sha256 or ""), target)
        if target.exists():
            if expected_sha256 and sha256_file(target) != expected_sha256:
                print(f"REPLACE {target}: sha256 mismatch")
            else:
                print(f"OK {target}")
                continue
        else:
            print(f"GET {target} <- {url}")
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_suffix(target.suffix + ".tmp")
        urllib.request.urlretrieve(url, temp_target)
        if sha256_file(temp_target) != expected_sha256:
            temp_target.unlink(missing_ok=True)
            raise SystemExit(f"sha256 mismatch after download: {target}")
        temp_target.replace(target)
    print(json.dumps({"manifest": str(manifest_path), "artifacts": len(artifacts)}, indent=2))
    return 0


def resolve_artifact_path(raw_path: str) -> Path:
    target = (ROOT / raw_path).resolve()
    models_root = (ROOT / "models").resolve()
    if models_root != target and models_root not in target.parents:
        raise SystemExit(f"refusing to write outside models/: {raw_path}")
    return target


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit(f"refusing non-HTTPS artifact URL: {url}")


def validate_sha256(value: str, target: Path) -> None:
    hex_chars = set("0123456789abcdefABCDEF")
    if len(value) != 64 or any(char not in hex_chars for char in value):
        raise SystemExit(f"artifact requires a 64-character sha256 before download: {target}")


if __name__ == "__main__":
    raise SystemExit(main())
