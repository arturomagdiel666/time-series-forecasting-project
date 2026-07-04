"""Precompute the Statistical Analysis block once into a committed artifact.

The dashboard's Statistical Analysis page used to run ADF, KPSS, Kruskal-Wallis
and Ljung-Box live over the full ~34k-row hourly series on every visit. That is
several seconds on Railway's small CPU and makes the page look broken. This
script computes those results a single time and writes them to
``reports/stats_results.json`` (committed, so the Railway deployment serves it),
letting the page just read the numbers.

Run: ``python scripts/precompute_stats.py``  (uses the same .venv, Python 3.12).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import stats_tests  # noqa: E402

# The statistical tests are deterministic, but pin the seed anyway so the whole
# project keeps a single reproducibility contract (seed=42 everywhere).
np.random.seed(42)

HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "household_power_hourly.parquet"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "stats_results.json"
TARGET = "Global_active_power"

# Approved figures from reports/report.md. The dashboard and the report must show
# identical numbers, so we assert the freshly computed values match these within a
# tolerance before writing the artifact. rel tolerances are loose (5%) because the
# report rounds; p-value checks are threshold-based since several underflow to ~0.
EXPECTED = {
    "kw_hour_h": 8842.5,
    "kw_dow_h": 239.3,
    "ljung_box": {24: 50757.0, 168: 196057.0, 720: 593902.0},
}


def _check(results: dict) -> list[str]:
    """Return a list of human-readable mismatches against the approved figures."""
    problems: list[str] = []

    kw_hour = results["kruskal"]["hour"]["statistic"]
    kw_dow = results["kruskal"]["day_of_week"]["statistic"]
    if not np.isclose(kw_hour, EXPECTED["kw_hour_h"], rtol=0.05):
        problems.append(f"Kruskal hour H={kw_hour:.1f} != {EXPECTED['kw_hour_h']}")
    if not np.isclose(kw_dow, EXPECTED["kw_dow_h"], rtol=0.05):
        problems.append(f"Kruskal dow H={kw_dow:.1f} != {EXPECTED['kw_dow_h']}")

    raw = results["stationarity"]["raw"]
    diff = results["stationarity"]["seasonal_diff"]
    # Raw: ADF rejects a unit root (p small) but KPSS rejects stationarity (p<=0.01).
    if raw["adf_pvalue"] > 1e-3:
        problems.append(f"raw ADF p={raw['adf_pvalue']:.2e} not << 0.05")
    if raw["kpss_pvalue"] > 0.05:
        problems.append(f"raw KPSS p={raw['kpss_pvalue']:.3f} not <= 0.05")
    if raw["verdict"] != "inconclusive":
        problems.append(f"raw verdict={raw['verdict']} (expected inconclusive)")
    # Seasonal difference: stationary under both tests.
    if diff["adf_pvalue"] > 0.05:
        problems.append(f"diff ADF p={diff['adf_pvalue']:.2e} not < 0.05")
    if diff["kpss_pvalue"] < 0.10:
        problems.append(f"diff KPSS p={diff['kpss_pvalue']:.3f} not >= 0.10")
    if diff["verdict"] != "stationary":
        problems.append(f"diff verdict={diff['verdict']} (expected stationary)")

    for row in results["ljung_box"]["results"]:
        lag, stat, pvalue = row["lag"], row["statistic"], row["pvalue"]
        expected_stat = EXPECTED["ljung_box"][lag]
        if not np.isclose(stat, expected_stat, rtol=0.05):
            problems.append(f"Ljung-Box lag {lag} Q={stat:.0f} != {expected_stat}")
        if pvalue > 1e-3:
            problems.append(f"Ljung-Box lag {lag} p={pvalue:.2e} not < 1e-3")

    return problems


def main() -> None:
    series = pd.read_parquet(HOURLY_PATH)[TARGET]
    results = stats_tests.summarize(series)

    # Echo the headline numbers so a human can eyeball them against the report.
    kw = results["kruskal"]
    raw = results["stationarity"]["raw"]
    diff = results["stationarity"]["seasonal_diff"]
    print(f"n_obs               : {results['n_obs']}")
    print(f"Kruskal hour  H/p   : {kw['hour']['statistic']:.1f} / {kw['hour']['pvalue']:.2e}")
    print(f"Kruskal dow   H/p   : {kw['day_of_week']['statistic']:.1f} / {kw['day_of_week']['pvalue']:.2e}")
    print(f"ADF  raw/diff p     : {raw['adf_pvalue']:.2e} / {diff['adf_pvalue']:.2e}")
    print(f"KPSS raw/diff p     : {raw['kpss_pvalue']:.3f} / {diff['kpss_pvalue']:.3f}")
    print(f"verdict raw/diff    : {raw['verdict']} / {diff['verdict']}")
    for row in results["ljung_box"]["results"]:
        print(f"Ljung-Box lag {row['lag']:>3} : Q={row['statistic']:.0f} p={row['pvalue']:.2e}")

    problems = _check(results)
    if problems:
        print("\nCROSS-CHECK FAILED — dashboard would disagree with the report:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nCross-check passed. Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
