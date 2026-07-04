"""Smoke test that the Streamlit dashboard boots without raising.

Uses the headless AppTest harness and the committed processed artifacts (no raw
data, no browser), so it runs the same way in CI.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


def test_dashboard_boots_without_exception() -> None:
    """The default page renders with no uncaught exception."""
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception


def test_statistical_analysis_page_renders_from_artifact() -> None:
    """The Statistical Analysis page renders (from the precomputed artifact)."""
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    # Navigate to the stats page via the sidebar radio, then re-run.
    app.radio[0].set_value("Statistical Analysis 📊").run()
    assert not app.exception
    body = " ".join(md.value for md in app.markdown)
    # Content that only appears once the precomputed results are rendered.
    assert "Kruskal-Wallis by hour" in body
    assert "Ljung-Box" in body
