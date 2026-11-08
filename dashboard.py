"""
OmniRetail - Interactive Executive Dashboard
Built with Streamlit.
Provides visual analytics, KPI cards, conversion funnels, heatmaps, and session data.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="OmniRetail - Store Attention & Conversion Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output_analytics"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"

# Custom Styling for polished dashboard look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #0066cc;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .sub-note {
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)


def load_kpis():
    kpi_file = OUTPUT_DIR / "zone_kpis.json"
    if kpi_file.exists():
        with open(kpi_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_sessions():
    csv_file = OUTPUT_DIR / "shopper_sessions.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return None


def main():
    st.title("🛒 OmniRetail: Shopper Attention & Conversion Intelligence")
    st.caption("Powered by **Ultralytics YOLO26 & YOLO26-Pose** | CCTV-based In-Store Behavioral Analytics")

    # Sidebar
    st.sidebar.header("🕹️ Control Panel")
    kpis = load_kpis()
    sessions = load_sessions()

    if not kpis or sessions is None:
        st.warning("⚠️ No analytics reports found in `data/output_analytics/`. Please run `python main.py` first to process a video!")
        st.info("Run in terminal:\n```bash\ncd C:\\Users\\malig\\Desktop\\OmniRetail\npython main.py --max-frames 600\n```")
        return

    # Compute Pick vs Put-back summary from sessions if available
    total_picks = int(sessions["items_picked"].sum()) if "items_picked" in sessions.columns else kpis.get("shelf_interactors", 0)
    total_putbacks = int(sessions["items_put_back"].sum()) if "items_put_back" in sessions.columns else 0
    net_taken = int(sessions["net_items_taken"].sum()) if "net_items_taken" in sessions.columns else int(sessions.get("items_taken_count", pd.Series([0])).sum())

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2b5c8f;">
            <div class="metric-title">Total Footfall</div>
            <div class="metric-value">{kpis['total_footfall']}</div>
            <div class="sub-note">Unique Shoppers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #0088cc;">
            <div class="metric-title">Shelf Picks</div>
            <div class="metric-value">{total_picks}</div>
            <div class="sub-note">Lifted off Shelf</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
