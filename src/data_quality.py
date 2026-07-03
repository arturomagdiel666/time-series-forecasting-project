"""Data quality audit for the raw minute-level series.

Findings drive the cleaning strategy: gap sizes decide fill vs interpolate,
and range violations flag sensor faults that must not leak into features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Plausible operating envelope for the French low-voltage grid and meter.
VOLTAGE_MIN = 220.0
VOLTAGE_MAX = 250.0
EXPECTED_FREQ = "1min"


@dataclass
class QualityReport:
    """Structured summary of a single quality audit."""

    n_rows: int
    missing_per_column: pd.Series
    missing_rate: float
    n_duplicated_timestamps: int
    range_violations: dict[str, int]
    n_gaps: int
    largest_gap_minutes: float
    expected_rows: int
    coverage: float
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a compact human-readable audit summary."""
        lines = [
            f"rows: {self.n_rows:,}",
            f"missing rate: {self.missing_rate:.4%}",
            f"duplicated timestamps: {self.n_duplicated_timestamps}",
            f"timestamp gaps: {self.n_gaps} (largest {self.largest_gap_minutes:.0f} min)",
            f"coverage vs expected grid: {self.coverage:.4%}",
            f"range violations: {self.range_violations}",
        ]
        return "\n".join(lines)


def missing_report(frame: pd.DataFrame) -> pd.Series:
    """Return per-column missing counts, descending."""
    return frame.isna().sum().sort_values(ascending=False)


def duplicate_timestamps(frame: pd.DataFrame) -> int:
    """Count rows sharing an already-seen index timestamp."""
    return int(frame.index.duplicated().sum())


def range_violations(frame: pd.DataFrame) -> dict[str, int]:
    """Count physically implausible values (out-of-band voltage, negative power)."""
    violations: dict[str, int] = {}
    voltage = frame["Voltage"]
    violations["voltage_out_of_band"] = int(
        ((voltage < VOLTAGE_MIN) | (voltage > VOLTAGE_MAX)).sum()
    )
    for col in ("Global_active_power", "Global_reactive_power", "Global_intensity"):
        violations[f"{col}_negative"] = int((frame[col] < 0).sum())
    return violations


def timestamp_continuity(frame: pd.DataFrame, freq: str = EXPECTED_FREQ) -> dict[str, float]:
    """Measure how far the index departs from a regular ``freq`` grid."""
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq=freq)
    deltas = frame.index.to_series().diff().dropna()
    step = pd.Timedelta(freq)
    gaps = deltas[deltas > step]
    largest = gaps.max().total_seconds() / 60 if not gaps.empty else 0.0
    return {
        "expected_rows": len(expected),
        "n_gaps": int(len(gaps)),
        "largest_gap_minutes": float(largest),
        "coverage": float(len(frame) / len(expected)) if len(expected) else 0.0,
    }


def audit(frame: pd.DataFrame) -> QualityReport:
    """Run the full audit and return a populated report."""
    per_col = missing_report(frame)
    total_cells = frame.size
    continuity = timestamp_continuity(frame)

    return QualityReport(
        n_rows=len(frame),
        missing_per_column=per_col,
        missing_rate=float(per_col.sum() / total_cells) if total_cells else 0.0,
        n_duplicated_timestamps=duplicate_timestamps(frame),
        range_violations=range_violations(frame),
        n_gaps=continuity["n_gaps"],
        largest_gap_minutes=continuity["largest_gap_minutes"],
        expected_rows=int(continuity["expected_rows"]),
        coverage=continuity["coverage"],
    )
