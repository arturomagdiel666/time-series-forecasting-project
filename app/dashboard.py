"""Streamlit dashboard for the household power forecasting project (Phase 17).

Reads only committed processed artifacts (parquets, serialized models, metadata
and figures) so it runs identically on a machine without the raw dataset, such
as the Railway deployment. All paths resolve from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import metrics, modeling, stats_tests  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
HOURLY_PATH = PROCESSED / "household_power_hourly.parquet"
FEATURES_PATH = PROCESSED / "household_power_hourly_features.parquet"
PREDICTIONS_DIR = PROCESSED / "predictions"
EDA_FIGURES = PROJECT_ROOT / "reports" / "figures" / "eda"
MODEL_FIGURES = PROJECT_ROOT / "reports" / "figures" / "modeling"
REPORT_PATH = PROJECT_ROOT / "reports" / "report.md"

TARGET = "Global_active_power"

# Every persisted model forecasts the next hour except the direct h=24 model.
HORIZONS = {
    "naive_lag1": "h=1",
    "naive_lag24": "h=1",
    "naive_lag168": "h=1",
    "sarima": "h=1",
    "lstm": "h=1",
    "xgboost": "h=1",
    "xgboost_tuned": "h=1",
    "xgboost_h24": "h=24",
}

PAGES = [
    "Home 🏠",
    "Exploration 🔍",
    "Statistical Analysis 📊",
    "Model Comparison ⚖️",
    "Forecasting Playground 🎮",
    "Scientific Report 📄",
    "Video Explanation 🎬",
]


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_hourly() -> pd.DataFrame:
    return pd.read_parquet(HOURLY_PATH)


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_data(show_spinner=False)
def load_predictions() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted(PREDICTIONS_DIR.glob("*.parquet"))}


@st.cache_resource(show_spinner=False)
def load_metadata() -> dict[str, object]:
    return modeling.load_metadata()


@st.cache_resource(show_spinner=False)
def load_h1_model() -> object:
    return modeling.load_model(modeling.BEST_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_h24_model() -> object:
    return modeling.load_model(modeling.BEST_MODEL_H24_PATH)


@st.cache_data(show_spinner=False)
def load_test_matrices() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return the next-hour test design, its target and the h=24 test design."""
    df = load_features()
    matrix = modeling.build_model_matrix(df, verbose=False)
    matrix_h24 = modeling.build_h24_matrix(df)
    return matrix.X_test, matrix.y_test, matrix_h24.X_test


@st.cache_data(show_spinner=False)
def compute_stats() -> dict[str, float]:
    """Recompute the Block A hypothesis statistics (cached; STL reuses figures)."""
    series = load_hourly()[TARGET]
    kw_hour = stats_tests.kruskal_by(series, "hour")
    kw_dow = stats_tests.kruskal_by(series, "day_of_week")
    stationarity = stats_tests.stationarity_raw_and_differenced(series)
    return {
        "kw_hour_h": kw_hour.statistic,
        "kw_hour_p": kw_hour.pvalue,
        "kw_dow_h": kw_dow.statistic,
        "kw_dow_p": kw_dow.pvalue,
        "raw": stationarity["raw"],
        "diff": stationarity["seasonal_diff"],
    }


@st.cache_data(show_spinner=False)
def leaderboard() -> pd.DataFrame:
    """Assemble the labelled test leaderboard from the prediction files."""
    scale = float(load_metadata()["mase_denominator"])
    rows = {}
    for name, frame in load_predictions().items():
        scores = metrics.evaluate(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), scale
        )
        scores["horizon"] = HORIZONS.get(name, "h=1")
        rows[name] = scores
    table = pd.DataFrame(rows).T[["horizon", "mae", "rmse", "mape", "smape", "mase"]]
    return table.sort_values(["horizon", "mae"])


def show_image(path: Path, caption: str) -> None:
    """Render a figure if it exists; otherwise a tidy placeholder."""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not available yet: {path.name}")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.title("Household Electricity Consumption Forecasting 🏠")
    st.write(
        "Short-horizon forecasting of household electricity demand from the UCI "
        "Individual Household Electric Power Consumption dataset, comparing naive "
        "baselines, SARIMA, an LSTM and gradient-boosted trees under a strict "
        "chronological evaluation."
    )

    meta = load_metadata()
    h1, h24 = meta["metrics"]["h1"], meta["metrics"]["h24"]
    st.subheader("Headline results (tuned XGBoost)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next-hour MAE", f"{h1['mae']:.3f} kWh")
    c2.metric("Next-hour MASE", f"{h1['mase']:.3f}")
    c3.metric("Next-day MAE", f"{h24['mae']:.3f} kWh")
    c4.metric("Next-day MASE", f"{h24['mase']:.3f}")

    st.subheader("Business questions")
    st.markdown(
        "1. Can we reliably forecast next-hour and next-day consumption?\n"
        "2. What are the dominant temporal patterns (hourly, daily, weekly, seasonal)?\n"
        "3. Which paradigm offers the best performance/complexity trade-off?"
    )

    hourly = load_hourly()
    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hourly rows", f"{len(hourly):,}")
    c2.metric("Start", str(hourly.index.min().date()))
    c3.metric("End", str(hourly.index.max().date()))

    st.subheader("Tech stack")
    st.markdown(
        "- pandas, NumPy, statsmodels, scikit-learn, XGBoost, PyTorch\n"
        "- MLflow tracking, pytest + GitHub Actions CI\n"
        "- Streamlit + Plotly dashboard, deployed on Railway"
    )


def page_exploration() -> None:
    st.title("Exploration 🔍")
    hourly = load_hourly()
    series = hourly[TARGET]

    lo, hi = series.index.min().to_pydatetime(), series.index.max().to_pydatetime()
    start, end = st.slider(
        "Date range", min_value=lo, max_value=hi, value=(lo, hi), format="YYYY-MM-DD"
    )
    window = series.loc[start:end]

    fig = px.line(window, labels={"value": "kWh", "timestamp": "time"})
    fig.update_traces(line_width=0.7)
    fig.update_layout(showlegend=False, title="Global active power")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        hist = px.histogram(window, nbins=60, title="Distribution")
        hist.update_layout(showlegend=False)
        st.plotly_chart(hist, use_container_width=True)
    with col2:
        roll = pd.DataFrame(
            {
                "24h mean": window.rolling(24).mean(),
                "168h mean": window.rolling(168).mean(),
                "168h std": window.rolling(168).std(),
            }
        )
        st.plotly_chart(
            px.line(roll, title="Rolling statistics"), use_container_width=True
        )

    frame = pd.DataFrame(
        {
            "value": window.to_numpy(),
            "hour": window.index.hour,
            "day_of_week": window.index.dayofweek,
            "month": window.index.month,
        }
    )
    grouping = st.selectbox("Boxplot by", ["hour", "day_of_week", "month"])
    st.plotly_chart(
        px.box(frame, x=grouping, y="value", labels={"value": "kWh"}),
        use_container_width=True,
    )


def page_stats() -> None:
    st.title("Statistical Analysis 📊")
    stats = compute_stats()

    st.subheader("H1 - Weekly (and daily) seasonality")
    st.markdown(
        f"Kruskal-Wallis by hour: **H = {stats['kw_hour_h']:.0f}**, "
        f"p = {stats['kw_hour_p']:.1e}. "
        f"By day-of-week: **H = {stats['kw_dow_h']:.0f}**, p = {stats['kw_dow_p']:.1e}. "
        "Both reject equal distributions, so daily and weekly seasonality are "
        "significant. **H1 supported.**"
    )

    st.subheader("H2 - Stationarity")
    raw, diff = stats["raw"], stats["diff"]
    table = pd.DataFrame(
        {
            "series": ["raw", "seasonal diff (lag-24)"],
            "ADF p-value": [raw.adf_pvalue, diff.adf_pvalue],
            "KPSS p-value": [raw.kpss_pvalue, diff.kpss_pvalue],
            "verdict": [raw.verdict, diff.verdict],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown(
        "The raw series is not cleanly stationary (ADF and KPSS disagree); the "
        "lag-24 seasonal difference is stationary under both tests. **H2 supported.**"
    )

    st.subheader("Autocorrelation and decomposition")
    show_image(EDA_FIGURES / "05_acf_pacf.png", "ACF and PACF (up to 168 lags)")
    col1, col2 = st.columns(2)
    with col1:
        show_image(EDA_FIGURES / "06_stl_daily.png", "STL decomposition (daily)")
    with col2:
        show_image(EDA_FIGURES / "07_stl_weekly.png", "STL decomposition (weekly)")


def page_comparison() -> None:
    st.title("Model Comparison ⚖️")
    board = leaderboard()

    st.subheader("Leaderboard (test set)")
    st.dataframe(board.round(4), use_container_width=True)

    metric = st.selectbox("Metric", ["mae", "rmse", "mape", "smape", "mase"])
    bar = px.bar(
        board.sort_values(metric),
        x=board.sort_values(metric).index,
        y=metric,
        color="horizon",
        title=f"{metric.upper()} by model",
    )
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("Actual vs predicted")
    preds = load_predictions()
    models = st.multiselect(
        "Models", list(preds.keys()), default=["xgboost_tuned", "sarima", "lstm"]
    )
    ref = preds["xgboost_tuned"]
    weeks = list(pd.date_range(ref.index.min().normalize(), ref.index.max(), freq="7D"))
    # Default to the grid week nearest the highest-demand winter week.
    default_week = min(weeks, key=lambda w: abs(w - pd.Timestamp("2010-02-08")))
    week_start = st.select_slider("Week start", options=weeks, value=default_week)
    end = week_start + pd.Timedelta(days=7)

    fig = go.Figure()
    truth = ref.loc[week_start:end, "y_true"]
    fig.add_trace(go.Scatter(x=truth.index, y=truth, name="actual", line=dict(color="black")))
    for name in models:
        seg = preds[name].loc[week_start:end, "y_pred"]
        fig.add_trace(go.Scatter(x=seg.index, y=seg, name=name, opacity=0.8))
    fig.update_layout(title=f"Week of {pd.Timestamp(week_start).date()}", yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residual diagnostics")
    which = st.selectbox("Model", ["xgboost_tuned", "sarima", "lstm"])
    show_image(MODEL_FIGURES / f"02_residuals_{which}.png", f"Residuals: {which}")
    show_image(MODEL_FIGURES / "03_error_breakdowns.png", "MAE by hour, day-of-week, month")


def page_playground() -> None:
    st.title("Forecasting Playground 🎮")
    st.write(
        "Pick a timestamp in the test period; the tuned models forecast that hour "
        "(next-hour) and the same hour one day ahead (next-day). Feature values come "
        "from the historical feature matrix, so they stay within the trained bounds."
    )

    x_test, y_test, x_test_h24 = load_test_matrices()
    model_h1, model_h24 = load_h1_model(), load_h24_model()

    lo, hi = y_test.index.min().to_pydatetime(), y_test.index.max().to_pydatetime()
    default = pd.Timestamp("2010-02-08 19:00:00").to_pydatetime()
    picked = st.slider(
        "Target timestamp",
        min_value=lo,
        max_value=hi,
        value=default,
        step=pd.Timedelta(hours=1).to_pytimedelta(),
        format="YYYY-MM-DD HH:mm",
    )
    tau = pd.Timestamp(picked)
    if tau not in y_test.index:
        tau = y_test.index[y_test.index.get_indexer([tau], method="nearest")[0]]

    actual = float(y_test.loc[tau])
    pred_h1 = float(model_h1.predict(x_test.loc[[tau]])[0])
    pred_h24 = float(model_h24.predict(x_test_h24.loc[[tau]])[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual", f"{actual:.3f} kWh")
    c2.metric("Next-hour forecast", f"{pred_h1:.3f} kWh", f"{pred_h1 - actual:+.3f}")
    c3.metric("Next-day forecast", f"{pred_h24:.3f} kWh", f"{pred_h24 - actual:+.3f}")

    context = y_test.loc[tau - pd.Timedelta(hours=48) : tau + pd.Timedelta(hours=6)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=context.index, y=context, name="actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=[tau], y=[pred_h1], name="next-hour", mode="markers",
                             marker=dict(size=12, symbol="circle")))
    fig.add_trace(go.Scatter(x=[tau], y=[pred_h24], name="next-day", mode="markers",
                             marker=dict(size=12, symbol="diamond")))
    fig.update_layout(title=f"Context around {tau}", yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)


def page_report() -> None:
    st.title("Scientific Report 📄")
    if REPORT_PATH.exists():
        st.markdown(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info(
            "The full scientific report is produced in Block D and will render here "
            "once `reports/report.md` is available."
        )


def page_video() -> None:
    st.title("Video Explanation 🎬")
    video_files = sorted((PROJECT_ROOT / "video").glob("*.mp4"))
    if video_files:
        st.video(str(video_files[0]))
    else:
        st.info("A walkthrough video will be added here. Coming soon.")


PAGE_FUNCS = {
    "Home 🏠": page_home,
    "Exploration 🔍": page_exploration,
    "Statistical Analysis 📊": page_stats,
    "Model Comparison ⚖️": page_comparison,
    "Forecasting Playground 🎮": page_playground,
    "Scientific Report 📄": page_report,
    "Video Explanation 🎬": page_video,
}


def main() -> None:
    st.set_page_config(page_title="Power Forecasting", layout="wide")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Page", PAGES, key="nav")
    st.sidebar.caption("Household electricity consumption forecasting")
    PAGE_FUNCS[choice]()


if __name__ == "__main__":
    main()
