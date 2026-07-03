"""CLI entry point to fetch and verify the raw dataset.

Run once to bootstrap a clean machine; normal pipeline runs read the local
file directly. Kept thin so the download logic has a single home in
``src.data_loader``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DEFAULT_RAW_DIR, RAW_FILENAME, load_or_download_data


def main() -> int:
    """Trigger the hybrid loader and report the resulting frame shape."""
    frame = load_or_download_data()
    print(f"loaded {len(frame):,} rows into {DEFAULT_RAW_DIR / RAW_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
