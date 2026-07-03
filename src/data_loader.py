"""Hybrid dataset loader with integrity verification.

Prefers a locally provided file so the pipeline never depends on network
access during normal runs; the Kaggle fallback exists only for a clean
machine bootstrap. SHA-256 guards against silent corruption or a swapped
source that would invalidate every downstream result.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd

RAW_FILENAME = "household_power_consumption.txt"
KAGGLE_DATASET = "uciml/electric-power-consumption-data-set"

# SHA-256 of the canonical UCI file (2,075,259 rows). A mismatch means the
# source changed and the hardcoded expectation must be revisited deliberately.
EXPECTED_SHA256 = "4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Sentinels used by the source for missing measurements.
NA_VALUES = ["?", ""]

FLOAT_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, streamed to bound memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_integrity(path: Path, expected: str = EXPECTED_SHA256) -> None:
    """Raise if the file hash does not match the expected value."""
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _download_from_kaggle(target_dir: Path) -> Path:
    """Fetch the dataset via kagglehub and copy it into ``target_dir``."""
    # Imported lazily so the module works without kagglehub installed.
    import kagglehub

    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    source = next(cache_dir.rglob(RAW_FILENAME), None)
    if source is None:
        raise FileNotFoundError(
            f"{RAW_FILENAME} not present in Kaggle download at {cache_dir}"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / RAW_FILENAME
    shutil.copy2(source, destination)
    return destination


def load_or_download_data(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    verify: bool = True,
) -> pd.DataFrame:
    """Load the raw series, downloading from Kaggle only if absent locally.

    Combines ``Date`` and ``Time`` into a minute-resolution ``DatetimeIndex``
    and casts measurements to float32 to halve memory on the ~2M-row frame.
    """
    raw_dir = Path(raw_dir)
    raw_path = raw_dir / RAW_FILENAME

    if not raw_path.exists():
        raw_path = _download_from_kaggle(raw_dir)

    if verify:
        verify_integrity(raw_path)

    return _parse_raw(raw_path)


def _parse_raw(raw_path: Path) -> pd.DataFrame:
    """Parse the semicolon-separated file into an indexed float32 frame."""
    frame = pd.read_csv(
        raw_path,
        sep=";",
        na_values=NA_VALUES,
        dtype={col: "float32" for col in FLOAT_COLUMNS},
        low_memory=False,
    )

    timestamp = pd.to_datetime(
        frame["Date"] + " " + frame["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    frame = frame.drop(columns=["Date", "Time"])
    frame.index = pd.DatetimeIndex(timestamp, name="timestamp")
    return frame


if __name__ == "__main__":
    df = load_or_download_data()
    print(df.shape)
    print(df.dtypes)
    print(df.head())
