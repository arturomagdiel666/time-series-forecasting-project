"""Unit tests for the hybrid data loader.

All tests use the synthetic fixture or small in-memory blobs; the network
(kagglehub) path is never exercised.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src import data_loader as dl

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "synthetic_power_data.csv"


def test_parse_raw_schema_and_dtypes() -> None:
    """Parsed frame has the float32 meter columns and a datetime index."""
    frame = dl._parse_raw(FIXTURE_CSV)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.columns) == dl.FLOAT_COLUMNS
    assert all(str(frame[c].dtype) == "float32" for c in dl.FLOAT_COLUMNS)
    # The fixture injects "?" markers, which must parse to NaN.
    assert frame["Global_active_power"].isna().any()


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    """The streamed hash equals a direct hashlib digest."""
    blob = b"household power test blob\n" * 10
    path = tmp_path / "blob.txt"
    path.write_bytes(blob)
    assert dl.sha256_of(path) == hashlib.sha256(blob).hexdigest()


def test_verify_integrity_pass(tmp_path: Path) -> None:
    """A matching hash passes without raising."""
    blob = b"correct-content"
    path = tmp_path / "ok.txt"
    path.write_bytes(blob)
    dl.verify_integrity(path, hashlib.sha256(blob).hexdigest())


def test_verify_integrity_fail(tmp_path: Path) -> None:
    """A mismatched hash raises ValueError."""
    path = tmp_path / "bad.txt"
    path.write_bytes(b"unexpected-content")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dl.verify_integrity(path, "0" * 64)


def test_load_uses_local_path_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the raw file exists locally, the Kaggle download is never called."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(FIXTURE_CSV, raw_dir / dl.RAW_FILENAME)

    def _boom(_target: Path) -> Path:
        raise AssertionError("kagglehub download must not be called")

    monkeypatch.setattr(dl, "_download_from_kaggle", _boom)

    frame = dl.load_or_download_data(raw_dir=raw_dir, verify=False)
    assert list(frame.columns) == dl.FLOAT_COLUMNS
    assert len(frame) == 1000
