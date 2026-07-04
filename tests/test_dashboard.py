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


def test_model_comparison_page_includes_timesfm() -> None:
    """Model Comparison renders and the leaderboard includes the TimesFM baseline."""
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.radio[0].set_value("Model Comparison ⚖️").run()
    assert not app.exception
    # The leaderboard is the first dataframe on the page; both horizons must appear.
    board_index = list(app.dataframe[0].value.index)
    assert "timesfm" in board_index
    assert "timesfm_h24" in board_index


def test_video_explanation_page_renders() -> None:
    """The Video Explanation page renders (embedded video or graceful fallback)."""
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.radio[0].set_value("Video Explanation 🎬").run()
    assert not app.exception
