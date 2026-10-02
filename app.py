

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PJMW Energy Demand Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
DATA_NAME = "PJMW_hourly_cleaned_edit.csv"
MODEL_NAME = "tuned_xgb_model.pkl"

FEATURES = [
    "hour", "Dayofweek", "isweekend", "Is_Holiday", "month", "year",
    "lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
TARGET = "PJMW_MW"
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

NAVY = "#0b1f4a"
BLUE = "#1e6bff"

PAGES = [
    ("Dashboard", "dashboard"),
    ("Historical", "monitoring"),
    ("Forecast", "rocket_launch"),
    ("Load Patterns", "bar_chart"),
    ("Data", "database"),
]


def find_file(name: str) -> Path:
    for p in (BASE / name, BASE / "data" / name, BASE / "models" / name, Path.cwd() / name):
        if p.exists():
            return p
    st.error(f"Could not find **{name}**. Put it in the same folder as app.py.")
    st.stop()


def html(s: str) -> str:
    """Strip indentation/blank lines so Streamlit's markdown never sees a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


_uid = {"n": 0}


def uid(prefix: str = "g") -> str:
    _uid["n"] += 1
    return f"{prefix}{_uid['n']}"


# ──────────────────────────────────────────────────────────────────────────────
# Data / model / features
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    raw = pd.read_csv(find_file(DATA_NAME), parse_dates=["Datetime"])
    df = (
        raw.groupby("Datetime", as_index=False)
        .agg({
            TARGET: "mean", "year": "first", "month": "first", "day": "first",
            "hour": "first", "Dayofweek": "first", "isweekend": "first",
            "Is_Holiday": "first", "Season": "first",
        })
        .sort_values("Datetime")
        .reset_index(drop=True)
    )
    df["Is_Holiday"] = df["Is_Holiday"].astype(bool)
    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = find_file(MODEL_NAME)
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as TrainedModels.ipynb."""
    d = df.copy()
    s = d.set_index("Datetime")[TARGET]
    d["lag_1"] = d[TARGET].shift(1)
    d["lag_24"] = s.reindex(d["Datetime"] - pd.Timedelta(hours=24)).values
    d["lag_168"] = s.reindex(d["Datetime"] - pd.Timedelta(hours=168)).values
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["Dayofweek"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["Dayofweek"] / 7)
    d["rolling_24"] = d[TARGET].shift(1).rolling(24).mean()
    d["rolling_168"] = d[TARGET].shift(1).rolling(168).mean()
    d["Is_Holiday"] = d["Is_Holiday"].astype(int)
    return d.dropna().reset_index(drop=True)


@st.cache_data(show_spinner="Evaluating model…")
def evaluate_model(days: int = 30) -> dict:
    """One-step-ahead accuracy on the last N days (real lags, as in the notebook)."""
    df = load_data()
    model = load_model()
    feat = build_features(df)
    ev = feat[feat["Datetime"] > feat["Datetime"].max() - pd.Timedelta(days=days)].copy()
    ev["Predicted"] = model.predict(ev[FEATURES])
    y, p = ev[TARGET], ev["Predicted"]
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "r2": 1 - ss_res / ss_tot,
        "mae": float((y - p).abs().mean()),
        "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "mape": float(((y - p).abs() / y).mean() * 100),
        "frame": ev[["Datetime", TARGET, "Predicted"]].reset_index(drop=True),
        "days": days,
    }


def predict_one(model, row: dict) -> float:
    x = np.array([[row[f] for f in FEATURES]], dtype=np.float32)
    try:  # fast path (same result as model.predict, ~8x quicker for long horizons)
        return float(model.get_booster().inplace_predict(x)[0])
    except Exception:
        return float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])


@st.cache_data(show_spinner="Running forecast…")
def make_forecast(horizon_days: int) -> pd.DataFrame:
    """Recursive hour-by-hour forecast: each prediction feeds the next hour's lags."""
    df = load_data()
    model = load_model()
    vals = list(df[TARGET].values[-168:])
    last = df["Datetime"].iloc[-1]
    first = last + pd.Timedelta(hours=1)
    # forecast through the end of the Nth calendar day (so daily rows are full days)
    end = first.normalize() + pd.Timedelta(days=horizon_days) - pd.Timedelta(hours=1)
    times = pd.date_range(first, end, freq="h")
    hol = set(
        USFederalHolidayCalendar()
        .holidays(times[0].normalize(), times[-1].normalize())
        .date
    )

    out = []
    for t in times:
        row = {
            "hour": t.hour,
            "Dayofweek": t.dayofweek,
            "isweekend": int(t.dayofweek >= 5),
            "Is_Holiday": int(t.date() in hol),
            "month": t.month,
            "year": t.year,
            "lag_1": vals[-1],
            "lag_24": vals[-24],
            "lag_168": vals[-168],
            "rolling_24": float(np.mean(vals[-24:])),
            "rolling_168": float(np.mean(vals[-168:])),
            "hour_sin": np.sin(2 * np.pi * t.hour / 24),
            "hour_cos": np.cos(2 * np.pi * t.hour / 24),
            "dow_sin": np.sin(2 * np.pi * t.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * t.dayofweek / 7),
        }
        pred = predict_one(model, row)
        vals.append(pred)
        out.append((t, pred))
    return pd.DataFrame(out, columns=["Datetime", "Forecast_MW"])


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(1200px 500px at 85% -5%, #e2f0ff 0%, rgba(226,240,255,0) 60%),
        linear-gradient(180deg, #f5f9ff 0%, #eef5ff 100%);
    color: #0b1f4a;
}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], [data-testid="stToolbarActions"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

/* keep the sidebar open/close controls visible and usable */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important; z-index: 999990 !important;
}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button {
    background: #ffffff !important; border: 1px solid #d6e4f7 !important; border-radius: 12px !important;
    box-shadow: 0 6px 16px rgba(30,60,120,.18) !important;
}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] * { color: #1e6bff !important; }
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] button { opacity: 1 !important; visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] *, [data-testid="stSidebarHeader"] button * { color: #dbe6ff !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1560px; }

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"][aria-expanded="true"] { width: 290px !important; min-width: 290px !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #0d2250 0%, #0a1a40 55%, #081433 100%) !important;
}
section[data-testid="stSidebar"] * { color: #dbe6ff; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.brand { display:flex; align-items:center; gap:12px; margin: 4px 0 18px 4px; }
.brand-name { font-size: 30px; font-weight: 800; color:#fff !important; line-height:1; letter-spacing:.5px; }
.brand-sub { font-size: 17px; color:#c6d5f7 !important; margin-top:4px; }
.side-h { display:flex; align-items:center; gap:10px; font-weight:700; font-size:17px; color:#fff !important; margin: 14px 0 2px 2px; }
.side-h svg { flex:none; }
.side-p { font-size:14.5px; color:#b9c8ea !important; margin: 0 0 8px 2px; line-height:1.4; }
.side-hr { border:none; border-top:1px solid rgba(255,255,255,.12); margin: 6px 0 4px 0; }

/* nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    display:flex; justify-content:flex-start; text-align:left; gap:12px; border: none; box-shadow: none;
    border-radius: 12px; padding: 11px 16px; background: transparent; color: #cfdcfb;
}
section[data-testid="stSidebar"] .stButton > button > div,
section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] { justify-content:flex-start; text-align:left; }
section[data-testid="stSidebar"] .stButton > button p { font-size: 17px; font-weight: 500; color: inherit !important; }
section[data-testid="stSidebar"] .stButton > button span[data-testid="stIconMaterial"] { font-size: 24px; color: inherit !important; }
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,.07); color:#fff; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #2b7bff 0%, #1657e0 100%); color:#fff; font-weight:600;
    box-shadow: 0 6px 18px rgba(30,107,255,.35);
}
section[data-testid="stSidebar"] .stButton { margin-bottom: -8px; }

/* sliders */
section[data-testid="stSidebar"] [data-testid="stSlider"] label p { font-size:14.5px; color:#c6d5f7 !important; }
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
    background:#16306a; border-radius:8px; padding:2px 8px; color:#fff !important;
}
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color:#8da3d6 !important; }
.st-key-forecast_days div[role="slider"] { background:#ff8f80 !important; }

/* range text boxes */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] label p { font-size:13px; color:#c6d5f7 !important; }
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"],
section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background: rgba(255,255,255,.09) !important; border-radius: 10px !important; border-color: rgba(255,255,255,.20) !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input { color:#ffffff !important; background: transparent !important; font-weight:600; }
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button { background: rgba(255,255,255,.10) !important; color:#fff !important; border:none !important; }
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button * { color:#fff !important; }
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input:disabled { opacity:.45; }

.powered {
    margin-top: 18px; padding: 16px 16px; border-radius: 16px;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
    display:flex; gap:12px; align-items:flex-start;
}
.powered b { color:#fff !important; font-size:15.5px; }
.powered span { font-size:13.5px; color:#b9c8ea !important; line-height:1.45; display:block; margin-top:4px; }
.side-foot {
    margin-top: 26px; padding: 10px 6px 4px 6px; display:flex; align-items:center; gap:12px;
    background: radial-gradient(120% 140% at 100% 100%, rgba(0,200,190,.28), rgba(0,0,0,0) 60%);
    border-radius: 14px;
}
.side-foot span { font-size:13.5px; line-height:1.3; color:#dbe6ff !important; }

/* ── Hero ────────────────────────────────────────────────── */
.hero { position:relative; min-height: 190px; margin-bottom: 8px; overflow:hidden; border-radius: 18px; }
.hero-art { position:absolute; right:0; top:0; width:64%; height:100%;
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 42%);
            mask-image: linear-gradient(90deg, transparent 0%, #000 42%); }
.hero-art svg { width:100%; height:100%; }
.hero-text { position:relative; z-index:2; padding: 22px 4px 8px 6px; }
.eyebrow { font-size: 13.5px; letter-spacing: 1.6px; color:#3c4f7a; font-weight:500; text-transform:uppercase; }
.hero h1 { font-size: 46px; font-weight: 800; margin: 4px 0 6px 0; color:#0b1f4a; letter-spacing:-.5px; line-height:1.1; padding:0; }
.hero h1 .grad { background: linear-gradient(90deg,#1e7bff,#12b8ff); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub { font-size: 17px; color:#33466f; }
.hero-sub i { color:#8aa0c8; font-style:normal; margin: 0 10px; }
.hero-badges { position:absolute; right: 8px; top: 6px; z-index:3; text-align:right; }
.pill-green { display:inline-flex; align-items:center; gap:12px; padding: 10px 18px; border-radius: 16px;
    background: linear-gradient(135deg, rgba(190,245,215,.95), rgba(210,248,232,.9)); border:1px solid #b6ecd0;
    box-shadow: 0 6px 16px rgba(30,180,110,.12); text-align:left; }
.pill-green .dot { width:22px; height:22px; border-radius:50%; background:#12c26d; box-shadow: 0 0 0 5px rgba(18,194,109,.22); }
.pill-green b { display:block; font-size:16px; color:#0b3d2a; }
.pill-green small { font-size:13px; color:#3a5a4c; }
.pill-date { display:inline-flex; align-items:center; gap:10px; margin-top:12px; padding: 10px 18px; border-radius: 14px;
    background: rgba(255,255,255,.75); border: 1px solid #d9e6f8; font-weight:500; color:#1b2c52; font-size:15px; }
.greener { margin-top: 10px; font-size: 13.5px; font-style: italic; color:#33466f; }

/* ── KPI cards ───────────────────────────────────────────── */
.kpi { position:relative; border-radius: 20px; padding: 16px 16px 12px 16px; height: 152px; overflow:hidden;
    border: 1px solid rgba(255,255,255,.9); box-shadow: 0 8px 24px rgba(30,60,120,.08); display:flex; gap:14px; }
.kpi-blue   { background: linear-gradient(135deg, #eaf3ff 0%, #f4f9ff 100%); border-color:#d6e6fb; }
.kpi-green  { background: linear-gradient(135deg, #e4fbf2 0%, #f1fdf8 100%); border-color:#cdeee0; }
.kpi-red    { background: linear-gradient(135deg, #ffeceb 0%, #fff6f5 100%); border-color:#ffd9d6; }
.kpi-purple { background: linear-gradient(135deg, #f0ecff 0%, #f8f5ff 100%); border-color:#e2dbfb; }
.kpi-icon { flex:none; width:56px; height:56px; border-radius:50%; display:flex; align-items:center; justify-content:center; }
.kpi-blue .kpi-icon   { background:#d6e7ff; }
.kpi-green .kpi-icon  { background:#c8f1de; }
.kpi-red .kpi-icon    { background:#ffd6d3; }
.kpi-purple .kpi-icon { background:#e0d8ff; }
.kpi-label { font-size:15.5px; font-weight:600; color:#1b2c52; }
.kpi-value { font-size:29px; font-weight:800; color:#0b1f4a; margin: 2px 0 2px 0; letter-spacing:-.5px; }
.kpi-sub { font-size:13px; color:#4a5b82; white-space:nowrap; }
.kpi-sub .dn { color:#e5353b; font-weight:700; }
.kpi-sub .up { color:#12a05c; font-weight:700; }
.kpi-spark { position:absolute; left:14px; right:14px; bottom:8px; height:34px; }
.kpi-spark svg { width:100%; height:100%; }

/* ── Tabs ────────────────────────────────────────────────── */
[data-testid="stMain"] [role="tablist"] {
    background: #e8f0fb; border: 1px solid #d8e5f7; border-radius: 16px; padding: 6px; gap: 6px; margin-bottom: 14px;
    box-shadow: none;
}
[data-testid="stMain"] [data-testid="stTab"] {
    height: 46px; padding: 0 26px; border-radius: 12px; background: transparent; color:#1b2c52; border: none;
}
[data-testid="stMain"] [data-testid="stTab"]::before, [data-testid="stMain"] [data-testid="stTab"]::after { display:none !important; }
[data-testid="stMain"] [data-testid="stTab"] { border-bottom: none !important; }
[data-testid="stMain"] [data-testid="stTab"] p { font-size: 16px; font-weight: 600; color:#1b2c52; }
[data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] {
    background: linear-gradient(135deg,#2b7bff,#1657e0) !important; box-shadow: 0 6px 16px rgba(30,107,255,.3);
}
[data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] p,
[data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] span { color:#fff !important; }
[data-testid="stMain"] [role="tablist"] > div[aria-hidden="true"],
[data-testid="stMain"] [role="tablist"] + div[aria-hidden="true"] { display:none !important; }

/* ── Cards ───────────────────────────────────────────────── */
[class*="st-key-card_"] {
    background: rgba(255,255,255,.92); border: 1px solid #e3ecf8 !important; border-radius: 20px !important;
    box-shadow: 0 8px 26px rgba(30,60,120,.07); padding: 12px 18px 8px 18px;
}
.card-head { display:flex; align-items:center; gap:14px; }
.card-icon { width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.card-title { font-size:21px; font-weight:800; color:#0b1f4a; line-height:1.2; }
.card-sub { font-size:14.5px; color:#4a5b82; }
.date-pill { display:inline-flex; align-items:center; gap:10px; padding: 10px 16px; border-radius:12px;
    background:#eef4fd; border:1px solid #dbe6f7; font-size:14px; color:#1b2c52; white-space:nowrap; }
.st-key-dl_hist button, .st-key-dl_fc button, .st-key-dl_data button {
    background: linear-gradient(135deg,#4aa0ff,#2b7bff); color:#fff; border:none; border-radius:12px;
    padding: 8px 18px; font-weight:600; box-shadow: 0 6px 16px rgba(43,123,255,.28);
}
.st-key-dl_hist button p, .st-key-dl_fc button p, .st-key-dl_data button p { color:#fff !important; }

.mini { border-radius:16px; padding:14px 16px; background:#f4f8ff; border:1px solid #e0ebfa; }
.mini small { color:#4a5b82; font-size:13.5px; }
.mini b { display:block; font-size:24px; color:#0b1f4a; margin-top:2px; }
.mini span { font-size:13px; color:#6a7ba3; }
.note { font-size:13.5px; color:#4a5b82; background:#f4f8ff; border:1px solid #e0ebfa; border-radius:12px; padding:10px 14px; }

/* ── Footer ──────────────────────────────────────────────── */
.footer { margin-top: 14px; padding: 14px 22px; border-radius:16px; background: linear-gradient(90deg,#eaf3ff,#f2f8ff);
    border:1px solid #dce8f8; display:flex; justify-content:space-between; align-items:center; font-size:14px; color:#33466f; }
.footer .l { display:flex; align-items:center; gap:14px; }
.footer .sep { color:#a7b7d6; }
.footer .r { color:#33466f; }
</style>
""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Small HTML/SVG builders
# ──────────────────────────────────────────────────────────────────────────────
def icon_svg(kind: str, color: str, size: int = 30) -> str:
    paths = {
        "bolt": f'<path d="M13 2 4 14h6l-1 8 9-12h-6z" fill="{color}"/>',
        "bars": f'<rect x="3" y="12" width="4.5" height="9" rx="1.2" fill="{color}"/><rect x="9.8" y="6" width="4.5" height="15" rx="1.2" fill="{color}"/><rect x="16.5" y="9" width="4.5" height="12" rx="1.2" fill="{color}"/>',
        "peak": f'<path d="M4 17 10 11l4 4 6-8" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M15 7h5v5" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>',
        "db": f'<ellipse cx="12" cy="6" rx="8" ry="3.2" fill="{color}"/><path d="M4 6v6c0 1.8 3.6 3.2 8 3.2s8-1.4 8-3.2V6c0 1.8-3.6 3.2-8 3.2S4 7.8 4 6z" fill="{color}" opacity=".85"/><path d="M4 12v6c0 1.8 3.6 3.2 8 3.2s8-1.4 8-3.2v-6c0 1.8-3.6 3.2-8 3.2S4 13.8 4 12z" fill="{color}" opacity=".7"/>',
        "chart": f'<path d="M3 3v18h18" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round"/><path d="m7 15 4-5 3 3 5-7" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>',
        "pie": f'<path d="M12 3a9 9 0 1 0 9 9h-9z" fill="{color}"/><path d="M14 2.2A9 9 0 0 1 21.8 10H14z" fill="{color}" opacity=".6"/>',
        "cal": f'<rect x="3.5" y="5" width="17" height="15" rx="3" fill="none" stroke="{color}" stroke-width="2"/><path d="M3.5 10h17M8 3v4M16 3v4" stroke="{color}" stroke-width="2" stroke-linecap="round"/>',
        "bulb": f'<path d="M12 3a6 6 0 0 0-3.5 10.9V17h7v-3.1A6 6 0 0 0 12 3z" fill="{color}"/><rect x="9" y="18.5" width="6" height="2.2" rx="1" fill="{color}"/>',
        "leaf": f'<path d="M20 4C9 4 4 9.5 4 15.5c0 1.4.4 2.6 1 3.5C5.5 12 11 9 20 4z" fill="{color}"/><path d="M4.5 20C8 14 12 11 17 9" stroke="{color}" stroke-width="1.6" fill="none" stroke-linecap="round" opacity=".7"/>',
        "rocket": f'<path d="M12 2c3.5 2.2 5.4 5.6 5.4 9.3L15 14H9l-2.4-2.7C6.6 7.6 8.5 4.2 12 2z" fill="{color}"/><circle cx="12" cy="9.2" r="1.9" fill="#fff"/><path d="M9 14 6.5 18l3-.8zM15 14l2.5 4-3-.8z" fill="{color}" opacity=".75"/>',
        "sliders": f'<path d="M4 7h10M18 7h2M4 17h2M10 17h10" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/><circle cx="16" cy="7" r="2.4" fill="none" stroke="{color}" stroke-width="2"/><circle cx="8" cy="17" r="2.4" fill="none" stroke="{color}" stroke-width="2"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24">{paths[kind]}</svg>'


def sparkline(values, color: str, w: int = 300, h: int = 38) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return ""
    if len(v) > 70:
        idx = np.linspace(0, len(v) - 1, 70).astype(int)
        v = v[idx]
    lo, hi = v.min(), v.max()
    rng = (hi - lo) or 1.0
    xs = np.linspace(0, w, len(v))
    ys = (h - 4) - (v - lo) / rng * (h - 10)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    g = uid("sp")
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="{g}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{color}" stop-opacity=".30"/>'
        f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="0,{h} {pts} {w},{h}" fill="url(#{g})"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>'
    )


def pylon(cx: float, base: float, h: float, color: str = "#1f4463") -> str:
    w = h * 0.30
    top = base - h
    a1, a2 = base - h * 0.62, base - h * 0.86
    return (
        f'<g stroke="{color}" stroke-width="1.6" fill="none" stroke-linejoin="round">'
        f'<path d="M{cx - w / 2},{base} L{cx - 3},{a1} L{cx - 1.5},{top + 6} L{cx},{top} L{cx + 1.5},{top + 6} L{cx + 3},{a1} L{cx + w / 2},{base}"/>'
        f'<path d="M{cx - w * .95},{a1} H{cx + w * .95} M{cx - w * .7},{a2} H{cx + w * .7}"/>'
        f'<path d="M{cx - w / 2},{base} L{cx + 3},{a1} M{cx + w / 2},{base} L{cx - 3},{a1} M{cx - 3},{a1} L{cx + 1.5},{top + 6} M{cx + 3},{a1} L{cx - 1.5},{top + 6}" stroke-width="1"/>'
        f'</g>'
    )


def turbine(cx: float, base: float, h: float, rot: float = 0, color: str = "#ffffff") -> str:
    hub_y = base - h
    blades = "".join(
        f'<path d="M{cx},{hub_y} L{cx - 1.6},{hub_y - h * .46} L{cx + 1.6},{hub_y - h * .46}Z" '
        f'transform="rotate({rot + a} {cx} {hub_y})" fill="{color}"/>'
        for a in (0, 120, 240)
    )
    return (
        f'<path d="M{cx - 1.8},{base} L{cx - .8},{hub_y} L{cx + .8},{hub_y} L{cx + 1.8},{base}Z" fill="{color}" opacity=".95"/>'
        f'{blades}<circle cx="{cx}" cy="{hub_y}" r="2.2" fill="{color}"/>'
    )


def hero_svg() -> str:
    trees = "".join(
        f'<path d="M{x},{y} l-6,14 h12z" fill="{c}"/><path d="M{x},{y - 8} l-5,12 h10z" fill="{c}"/>'
        for x, y, c in [(330, 132, "#2f9e73"), (344, 128, "#3fb387"), (358, 134, "#2f9e73"),
                        (372, 130, "#49bd90"), (388, 136, "#2f9e73"), (300, 138, "#49bd90")]
    )
    return f"""
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMaxYMid slice">
<defs>
<linearGradient id="hsky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#cfe6ff"/><stop offset="1" stop-color="#f4faff"/></linearGradient>
<radialGradient id="hsun"><stop offset="0" stop-color="#fff0b8"/><stop offset="0.45" stop-color="#ffd76a"/><stop offset="1" stop-color="#ffd76a" stop-opacity="0"/></radialGradient>
<linearGradient id="hh1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#9adfb9"/><stop offset="1" stop-color="#6cc9a0"/></linearGradient>
<linearGradient id="hh2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4fbf98"/><stop offset="1" stop-color="#2f9e86"/></linearGradient>
</defs>
<rect width="640" height="190" fill="url(#hsky)"/>
<circle cx="250" cy="52" r="58" fill="url(#hsun)"/>
<circle cx="250" cy="52" r="24" fill="#ffd66b"/>
<g fill="#fff" opacity=".95">
<ellipse cx="160" cy="46" rx="34" ry="9"/><ellipse cx="184" cy="40" rx="22" ry="9"/><ellipse cx="140" cy="42" rx="18" ry="8"/>
<ellipse cx="392" cy="26" rx="30" ry="8"/><ellipse cx="412" cy="21" rx="18" ry="8"/>
<ellipse cx="60" cy="90" rx="30" ry="7" opacity=".8"/>
</g>
<path d="M0,190 L0,120 Q70,80 150,112 T300,104 T460,96 T640,110 L640,190Z" fill="url(#hh1)"/>
{turbine(70, 118, 46, 12)}{turbine(118, 112, 34, 40)}{turbine(180, 116, 40, 75)}
{pylon(455, 132, 122)}{pylon(585, 128, 128)}
<path d="M455,88 Q520,104 585,86" stroke="#1f4463" stroke-width="1.2" fill="none"/>
<path d="M0,190 L0,150 Q110,118 230,148 T450,146 T640,150 L640,190Z" fill="url(#hh2)"/>
{trees}
<g fill="#2b8c74"><circle cx="500" cy="152" r="6"/><circle cx="512" cy="155" r="5"/><circle cx="520" cy="150" r="6"/><circle cx="620" cy="150" r="6"/></g>
</svg>
"""


def card_head(icon: str, bg: str, color: str, title: str, sub: str) -> str:
    return html(
        f"""
<div class="card-head">
<div class="card-icon" style="background:{bg}">{icon_svg(icon, color, 26)}</div>
<div><div class="card-title">{title}</div><div class="card-sub">{sub}</div></div>
</div>"""
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plotly helpers
# ──────────────────────────────────────────────────────────────────────────────
PLOT_CFG = {"displayModeBar": False}


def style_fig(fig: go.Figure, height: int, xtitle: str | None = None, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12.5, color="#33415c"),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(title=xtitle, showgrid=False, linecolor="#cfd9ea", ticks="outside", tickcolor="#cfd9ea"),
        yaxis=dict(title=ytitle, gridcolor="rgba(120,140,180,.16)", zeroline=False),
    )
    return fig


@st.cache_data(show_spinner=False)
def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def downsample(period: pd.DataFrame, max_points: int = 6000) -> tuple[pd.DataFrame, str | None]:
    """Average long ranges so the line chart stays fast (raw data is untouched)."""
    n = len(period)
    if n <= max_points:
        return period, None
    for hrs, label in [(3, "3-hour"), (6, "6-hour"), (12, "12-hour"), (24, "daily"), (72, "3-day"), (168, "weekly")]:
        if n / hrs <= max_points:
            break
    out = period.set_index("Datetime")[TARGET].resample(f"{hrs}h").mean().dropna().reset_index()
    return out, f"Showing {label} averages for readability ({n:,} hourly points in range). Download gives the raw hourly data."


def fmt_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %d, %Y")


# ──────────────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────────────
def section_historical(period: pd.DataFrame) -> None:
    with st.container(key="card_hist"):
        c1, c2, c3 = st.columns([5.2, 3.2, 1.6], vertical_alignment="center")
        c1.markdown(
            card_head("chart", "#e3eeff", BLUE, "Historical Energy Demand", "Hourly power demand over the selected period"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div style="text-align:right"><span class="date-pill">{icon_svg("cal", BLUE, 18)}'
            f'{fmt_date(period["Datetime"].min())} &nbsp;-&nbsp; {fmt_date(period["Datetime"].max())}</span></div>',
            unsafe_allow_html=True,
        )
        c3.download_button(
            "Download",
            data=csv_bytes(period[["Datetime", TARGET]]),
            file_name="pjmw_historical_demand.csv",
            mime="text/csv",
            icon=":material/download:",
            key="dl_hist",
            width="stretch",
        )
        shown, ds_note = downsample(period)
        fig = go.Figure(
            go.Scatter(
                x=shown["Datetime"], y=shown[TARGET], mode="lines",
                line=dict(color="#0a66ff", width=1.5 if ds_note is None else 1.1),
                fill="tozeroy", fillcolor="rgba(60,150,255,.16)",
                hovertemplate="%{y:,.0f} MW<extra></extra>",
            )
        )
        style_fig(fig, 340, "Timestamp", "Demand (MW)")
        fig.update_yaxes(range=[0, float(shown[TARGET].max()) * 1.12])
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_hist")
        if ds_note:
            st.caption(ds_note)

    left, right = st.columns(2)
    with left, st.container(key="card_dist"):
        st.markdown(
            card_head("pie", "#ffe4ec", "#ff4d8d", "Demand Distribution", "Distribution of power demand (filtered)"),
            unsafe_allow_html=True,
        )
        counts, edges = np.histogram(period[TARGET], bins=40)
        centers = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure(
            go.Bar(
                x=centers, y=counts, width=(edges[1] - edges[0]) * 0.96,
                marker=dict(color=counts, colorscale=[[0, "#ffb98a"], [0.5, "#ff7a9c"], [1, "#ff3d81"]], line=dict(width=0)),
                hovertemplate="%{x:,.0f} MW: %{y} hrs<extra></extra>",
            )
        )
        style_fig(fig, 300, "Demand (MW)", "Count")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_dist")

    with right, st.container(key="card_dow"):
        st.markdown(
            card_head("cal", "#dcf7ec", "#16b07a", "Average Demand by Day of Week", "Typical demand pattern across the week"),
            unsafe_allow_html=True,
        )
        dow = period.groupby(period["Datetime"].dt.dayofweek)[TARGET].mean().reindex(range(7))
        fig = go.Figure(
            go.Bar(
                x=DOW_NAMES, y=dow.values,
                marker=dict(
                    color=["#12c2f5", "#22dba5", "#5fd97a", "#ffd23f", "#ffa94d", "#ff8a5c", "#ff5d8f"],
                    line=dict(width=0),
                ),
                width=0.72,
                hovertemplate="%{y:,.0f} MW<extra></extra>",
            )
        )
        style_fig(fig, 300, None, "Demand (MW)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_dow")


def section_forecast(df: pd.DataFrame, ev: dict, f_from: int, f_to: int) -> None:
    fc_all = make_forecast(f_to)
    first_day = fc_all["Datetime"].iloc[0].normalize()
    day_no = (fc_all["Datetime"].dt.normalize() - first_day).dt.days + 1
    fc = fc_all[day_no >= f_from].reset_index(drop=True)
    n_days = f_to - f_from + 1
    last_ts = df["Datetime"].iloc[-1]
    peak_row = fc.loc[fc["Forecast_MW"].idxmax()]
    low_row = fc.loc[fc["Forecast_MW"].idxmin()]

    with st.container(key="card_fc"):
        c1, c2 = st.columns([8, 1.7], vertical_alignment="center")
        c1.markdown(
            card_head("rocket", "#ffe9e3", "#ff6b4a",
                      f"{f_to}-Day Demand Forecast" if f_from == 1 else f"Demand Forecast — Day {f_from} to Day {f_to}",
                      f"Hourly forecast from {fmt_date(fc['Datetime'].min())} to {fmt_date(fc['Datetime'].max())}"),
            unsafe_allow_html=True,
        )
        c2.download_button(
            "Download",
            data=csv_bytes(fc),
            file_name=f"pjmw_forecast_day{f_from}-{f_to}.csv",
            mime="text/csv",
            icon=":material/download:",
            key="dl_fc",
            width="stretch",
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(html(f'<div class="mini"><small>Forecast average</small><b>{fc["Forecast_MW"].mean():,.0f} MW</b><span>over {n_days} day{"s" if n_days != 1 else ""}</span></div>'), unsafe_allow_html=True)
        m2.markdown(html(f'<div class="mini"><small>Forecast peak</small><b>{peak_row["Forecast_MW"]:,.0f} MW</b><span>{peak_row["Datetime"]:%a %b %d, %H:%M}</span></div>'), unsafe_allow_html=True)
        m3.markdown(html(f'<div class="mini"><small>Forecast low</small><b>{low_row["Forecast_MW"]:,.0f} MW</b><span>{low_row["Datetime"]:%a %b %d, %H:%M}</span></div>'), unsafe_allow_html=True)
        m4.markdown(html(f'<div class="mini"><small>Last actual</small><b>{df[TARGET].iloc[-1]:,.0f} MW</b><span>{last_ts:%b %d, %Y %H:%M}</span></div>'), unsafe_allow_html=True)

        recent = df[df["Datetime"] > last_ts - timedelta(days=7 if f_to <= 14 else 21)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["Datetime"], y=recent[TARGET], name="Actual", mode="lines",
                                 line=dict(color="#0a66ff", width=1.8), hovertemplate="%{y:,.0f} MW<extra>Actual</extra>"))
        joined = fc if f_from > 1 else pd.concat(
            [recent.tail(1).rename(columns={TARGET: "Forecast_MW"})[["Datetime", "Forecast_MW"]], fc])
        fig.add_trace(go.Scatter(x=joined["Datetime"], y=joined["Forecast_MW"], name="Forecast", mode="lines",
                                 line=dict(color="#ff6b4a", width=2, dash="dot"), fill="tozeroy",
                                 fillcolor="rgba(255,107,74,.08)", hovertemplate="%{y:,.0f} MW<extra>Forecast</extra>"))
        fig.add_shape(type="line", x0=last_ts, x1=last_ts, y0=0, y1=1, yref="paper", line=dict(color="#8aa0c8", width=1, dash="dash"))
        fig.add_annotation(x=last_ts, y=1, yref="paper", text="forecast starts", showarrow=False, xanchor="left", yanchor="top", font=dict(size=11, color="#6a7ba3"))
        style_fig(fig, 360, "Timestamp", "Demand (MW)")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
        fig.update_yaxes(range=[0, max(float(fc["Forecast_MW"].max()), float(recent[TARGET].max())) * 1.12])
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_fc")

        daily = (
            fc.assign(Date=fc["Datetime"].dt.date)
            .groupby("Date")["Forecast_MW"].agg(Average="mean", Peak="max", Low="min").round(0).astype(int).reset_index()
        )
        daily["Day"] = pd.to_datetime(daily["Date"]).dt.strftime("%a")
        daily.insert(0, "Day #", (pd.to_datetime(daily["Date"]) - first_day).dt.days + 1)
        st.dataframe(daily[["Day #", "Date", "Day", "Average", "Peak", "Low"]], hide_index=True, width="stretch",
                     height=min(38 * (len(daily) + 1), 380))
        st.markdown(
            '<div class="note">Forecast is recursive: each predicted hour feeds the next hour\'s lag / rolling features, '
            'so error grows with the horizon. In a backtest over the last year the average error was roughly '
            '3–5% on day 1, ~9% on days 2–14 and ~12% on days 15–30, and the model tends to under-predict by ~5% '
            'on long horizons — much higher than the ~1% one-step-ahead error behind the R² card. '
            'Treat 30+ day forecasts as a trend, not an hourly promise. Holidays use the US federal holiday calendar.</div>',
            unsafe_allow_html=True,
        )

    with st.container(key="card_eval"):
        st.markdown(
            card_head("chart", "#efe9ff", "#7a5cff", "Model Check — Actual vs Predicted",
                      f"Chart: last 7 days · Metrics: one-step-ahead, last {ev['days']} days — R² {ev['r2']:.3f} · MAE {ev['mae']:,.0f} MW · MAPE {ev['mape']:.2f}%"),
            unsafe_allow_html=True,
        )
        fr = ev["frame"]
        fr = fr[fr["Datetime"] > fr["Datetime"].max() - timedelta(days=7)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fr["Datetime"], y=fr[TARGET], name="Actual", line=dict(color="#0a66ff", width=1.8)))
        fig.add_trace(go.Scatter(x=fr["Datetime"], y=fr["Predicted"], name="Predicted", line=dict(color="#ff6b4a", width=1.6, dash="dot")))
        style_fig(fig, 300, "Timestamp", "Demand (MW)")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_eval")


def section_patterns(df: pd.DataFrame, period: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left, st.container(key="card_heat"):
        st.markdown(
            card_head("bars", "#e3eeff", BLUE, "Hour × Weekday Heatmap", "Average demand in the selected period"),
            unsafe_allow_html=True,
        )
        piv = (
            period.assign(dow=period["Datetime"].dt.dayofweek, hr=period["Datetime"].dt.hour)
            .pivot_table(index="hr", columns="dow", values=TARGET, aggfunc="mean")
            .reindex(index=range(24), columns=range(7))
        )
        fig = go.Figure(
            go.Heatmap(
                z=piv.values, x=DOW_NAMES, y=list(range(24)),
                colorscale=[[0, "#eaf3ff"], [0.5, "#6fb1ff"], [1, "#0a3fbf"]],
                colorbar=dict(title="MW", thickness=12),
                hovertemplate="%{x} %{y}:00 → %{z:,.0f} MW<extra></extra>",
            )
        )
        style_fig(fig, 420, None, "Hour of day")
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(autorange="reversed", dtick=3)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_heat")

    with right, st.container(key="card_season"):
        st.markdown(
            card_head("cal", "#dcf7ec", "#16b07a", "Daily Load Shape by Season", "Average hourly demand across all history"),
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        colors = {"Winter": "#3b82f6", "Spring": "#22c55e", "Summer": "#ff6b4a", "Fall": "#f5a524"}
        for season, col in colors.items():
            g = df[df["Season"] == season].groupby("hour")[TARGET].mean()
            fig.add_trace(go.Scatter(x=g.index, y=g.values, name=season, mode="lines", line=dict(color=col, width=2.6),
                                     hovertemplate="%{y:,.0f} MW<extra>" + season + "</extra>"))
        style_fig(fig, 420, "Hour of day", "Demand (MW)")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_season")

    with st.container(key="card_month"):
        st.markdown(
            card_head("chart", "#fff0d9", "#f59e0b", "Monthly Average Demand", "Average demand per calendar month across all history"),
            unsafe_allow_html=True,
        )
        mo = df.groupby("month")[TARGET].mean().reindex(range(1, 13))
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = go.Figure(go.Bar(x=names, y=mo.values, marker=dict(color=mo.values, colorscale=[[0, "#8ec5ff"], [1, "#1e6bff"]]),
                               hovertemplate="%{y:,.0f} MW<extra></extra>"))
        style_fig(fig, 300, None, "Demand (MW)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_month")


def section_data(df: pd.DataFrame, period: pd.DataFrame, model) -> None:
    with st.container(key="card_data"):
        c1, c2 = st.columns([8, 1.7], vertical_alignment="center")
        c1.markdown(
            card_head("db", "#e0d8ff", "#7a5cff", "Dataset", f"{len(df):,} hourly rows · {fmt_date(df['Datetime'].min())} → {fmt_date(df['Datetime'].max())}"),
            unsafe_allow_html=True,
        )
        c2.download_button("Download", data=csv_bytes(period), file_name="pjmw_selected_period.csv",
                           mime="text/csv", icon=":material/download:", key="dl_data", width="stretch")
        view = period.sort_values("Datetime", ascending=False)[["Datetime", TARGET, "hour", "Dayofweek", "Season", "Is_Holiday"]]
        st.dataframe(view.head(5000), hide_index=True, height=340, width="stretch")
        st.caption(
            f"Selected period: {len(period):,} rows."
            + (" Table shows the latest 5,000 — use Download for everything." if len(period) > 5000 else "")
        )

    left, right = st.columns(2)
    with left, st.container(key="card_stats"):
        st.markdown(card_head("bars", "#e3eeff", BLUE, "Summary Statistics", "Selected period"), unsafe_allow_html=True)
        st.dataframe(period[TARGET].describe().round(1).rename("PJMW_MW").to_frame(), width="stretch")
    with right, st.container(key="card_imp"):
        st.markdown(card_head("bolt", "#ffe9e3", "#ff6b4a", "Feature Importance", "Tuned XGBoost model"), unsafe_allow_html=True)
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker=dict(color=imp.values, colorscale=[[0, "#ffd0bf"], [1, "#ff6b4a"]])))
        style_fig(fig, 340, "Importance", None)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="fig_imp")


# ──────────────────────────────────────────────────────────────────────────────
# Page chrome
# ──────────────────────────────────────────────────────────────────────────────
def _fix_ranges() -> None:
    """Keep 'from' strictly below 'to' for both ranges."""
    ss = st.session_state
    ss.hist_from = max(0, min(int(ss.hist_from), int(ss.hist_to) - 1))
    ss.fc_from = max(1, min(int(ss.fc_from), int(ss.fc_to)))


def _hist_slider_changed() -> None:
    st.session_state.hist_to = st.session_state.hist_days
    _fix_ranges()


def _hist_box_changed() -> None:
    st.session_state.hist_days = st.session_state.hist_to
    _fix_ranges()


def _fc_slider_changed() -> None:
    st.session_state.fc_to = st.session_state.forecast_days
    _fix_ranges()


def _fc_box_changed() -> None:
    st.session_state.forecast_days = st.session_state.fc_to
    _fix_ranges()


def sidebar(total_days: int) -> tuple[tuple[int, int] | None, tuple[int, int]]:
    ss = st.session_state
    if "page" not in ss:
        ss.page = "Dashboard"
    for k, v in {"hist_days": 30, "hist_to": 30, "hist_from": 0,
                 "forecast_days": 30, "fc_to": 30, "fc_from": 1}.items():
        ss.setdefault(k, v)

    with st.sidebar:
        st.markdown(
            html(
                f"""
<div class="brand">
<svg width="46" height="52" viewBox="0 0 24 28"><defs><linearGradient id="bolt" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffb347"/><stop offset="1" stop-color="#ff5f2e"/></linearGradient></defs>
<path d="M15 1 3 16h7l-2 11L21 11h-7z" fill="url(#bolt)"/></svg>
<div><div class="brand-name">PJMW</div><div class="brand-sub">Energy Forecast</div></div>
</div>"""
            ),
            unsafe_allow_html=True,
        )
        for name, icon in PAGES:
            active = st.session_state.page == name
            if st.button(name, icon=f":material/{icon}:", key=f"nav_{name}",
                         type="primary" if active else "secondary", width="stretch"):
                st.session_state.page = name
                st.rerun()

        st.markdown('<hr class="side-hr">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="side-h">{icon_svg("sliders", "#5db3ff", 24)}Control Panel</div>'
            '<div class="side-p">Adjust the settings to explore data and generate forecasts.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="side-h">{icon_svg("cal", "#5db3ff", 22)}Historical Period</div>', unsafe_allow_html=True)
        show_all = st.checkbox(f"Show all data ({total_days:,} days)", value=False, key="show_all")
        st.slider("Show last N days", 1, total_days, key="hist_days", format="%d days",
                  disabled=show_all, on_change=_hist_slider_changed)
        st.markdown('<div class="side-p" style="margin-bottom:2px">Custom range — days back from the latest data</div>',
                    unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        h1.number_input("From day", min_value=0, max_value=total_days, step=1, key="hist_from",
                        disabled=show_all, on_change=_fix_ranges)
        h2.number_input("To day", min_value=1, max_value=total_days, step=1, key="hist_to",
                        disabled=show_all, on_change=_hist_box_changed)

        st.markdown(f'<div class="side-h">{icon_svg("rocket", "#ff9b8a", 22)}Forecast Horizon</div>', unsafe_allow_html=True)
        st.slider("Select forecast days", 1, 60, key="forecast_days", format="%d days",
                  on_change=_fc_slider_changed)
        st.markdown('<div class="side-p" style="margin-bottom:2px">Custom range — days ahead (day 1 = first forecast day)</div>',
                    unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        f1.number_input("From day", min_value=1, max_value=60, step=1, key="fc_from", on_change=_fix_ranges)
        f2.number_input("To day", min_value=1, max_value=60, step=1, key="fc_to", on_change=_fc_box_changed)

        st.markdown(
            html(
                f"""
<div class="powered">
{icon_svg("bulb", "#ffd166", 26)}
<div><b>Powered by<br>XGBoost (Tuned)</b>
<span>Uses a pre-trained model and bundled dataset.<br>No file upload required.</span></div>
</div>
<div class="side-foot">{icon_svg("leaf", "#3ddc97", 34)}<span>Smarter Energy<br>Brighter Tomorrow</span></div>"""
            ),
            unsafe_allow_html=True,
        )
    hist_range = None if show_all else (int(ss.hist_from), int(ss.hist_to))
    fc_range = (int(ss.fc_from), int(ss.fc_to))
    return hist_range, fc_range


def hero(last_ts) -> None:
    st.markdown(
        html(
            f"""
<div class="hero">
<div class="hero-art">{hero_svg()}</div>
<div class="hero-text">
<div class="eyebrow">Powering a sustainable tomorrow</div>
<h1>PJMW <span class="grad">Energy</span> Demand Forecast</h1>
<div class="hero-sub">Historical Analysis<i>|</i>Machine Learning Forecast<i>|</i>Data-Driven Insights</div>
</div>
<div class="hero-badges">
<div class="pill-green"><span class="dot"></span><div><b>Model Loaded</b><small>XGBoost (Tuned)</small></div></div><br>
<div class="pill-date">{icon_svg("cal", BLUE, 18)}{fmt_date(last_ts)}</div>
<div class="greener">for a Greener Grid {icon_svg("leaf", "#16b07a", 14)}</div>
</div>
</div>"""
        ),
        unsafe_allow_html=True,
    )


def kpis(df: pd.DataFrame, period: pd.DataFrame, ev: dict) -> None:
    latest, prev = float(df[TARGET].iloc[-1]), float(df[TARGET].iloc[-2])
    delta = (latest - prev) / prev * 100
    cls, arrow = ("up", "▲") if delta >= 0 else ("dn", "▼")
    daily_mean = period.set_index("Datetime")[TARGET].resample("D").mean().values
    daily_max = period.set_index("Datetime")[TARGET].resample("D").max().values
    ev_daily = ev["frame"].set_index("Datetime")[TARGET].resample("D").mean().values

    cards = [
        ("kpi-blue", "bolt", "#2f7dff", "Latest Demand", f"{latest:,.0f} MW",
         f'<span class="{cls}">{arrow} {delta:+.2f}%</span> vs previous hour',
         sparkline(df[TARGET].values[-72:], "#2f7dff")),
        ("kpi-green", "bars", "#16b07a", "Average Demand", f"{period[TARGET].mean():,.0f} MW",
         "Selected period average", sparkline(daily_mean, "#12b886")),
        ("kpi-red", "peak", "#ff5a4d", "Peak Demand", f"{period[TARGET].max():,.0f} MW",
         "Maximum observed", sparkline(daily_max, "#ff4d4d")),
        ("kpi-purple", "db", "#6d4aff", "Model R²", f"{ev['r2']:.3f}",
         f"Last {ev['days']} days evaluation", sparkline(ev_daily, "#7a5cff")),
    ]
    for col, (klass, ic, icol, label, value, sub, spark) in zip(st.columns(4), cards):
        col.markdown(
            html(
                f"""
<div class="kpi {klass}">
<div class="kpi-icon">{icon_svg(ic, icol, 30)}</div>
<div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>
<div class="kpi-spark">{spark}</div>
</div>"""
            ),
            unsafe_allow_html=True,
        )


def footer() -> None:
    st.markdown(
        html(
            f"""
<div class="footer">
<div class="l">{icon_svg("leaf", "#16b07a", 20)}<span>PJMW Energy Demand Forecast</span><span class="sep">|</span><span>Built with Streamlit</span><span class="sep">|</span><span>© {datetime.now().year}</span></div>
<div class="r">Clean Energy &nbsp;•&nbsp; Smarter Grids &nbsp;•&nbsp; Brighter Tomorrow &nbsp;→</div>
</div>"""
        ),
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    df = load_data()
    model = load_model()
    ev = evaluate_model(30)
    last_ts = df["Datetime"].iloc[-1]
    total_days = int((last_ts - df["Datetime"].iloc[0]).days)
    hist_range, fc_range = sidebar(total_days)

    if hist_range is None:  # "Show all data"
        period = df.copy()
    else:
        h_from, h_to = hist_range
        period = df[
            (df["Datetime"] > last_ts - timedelta(days=h_to))
            & (df["Datetime"] <= last_ts - timedelta(days=h_from))
        ].reset_index(drop=True)
        if period.empty:
            period = df.tail(24 * 30).reset_index(drop=True)

    hero(last_ts)
    kpis(df, period, ev)
    st.write("")

    page = st.session_state.page
    if page == "Dashboard":
        t1, t2, t3, t4 = st.tabs([
            ":material/show_chart: Historical",
            ":material/rocket_launch: Forecast",
            ":material/bar_chart: Load Patterns",
            ":material/database: Data",
        ])
        with t1:
            section_historical(period)
        with t2:
            section_forecast(df, ev, *fc_range)
        with t3:
            section_patterns(df, period)
        with t4:
            section_data(df, period, model)
    elif page == "Historical":
        section_historical(period)
    elif page == "Forecast":
        section_forecast(df, ev, *fc_range)
    elif page == "Load Patterns":
        section_patterns(df, period)
    elif page == "Data":
        section_data(df, period, model)

    footer()


main()