from __future__ import annotations
# pyright: reportMissingImports=false

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from transwithai_cohere_timestamp.cli import main as package_main

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
