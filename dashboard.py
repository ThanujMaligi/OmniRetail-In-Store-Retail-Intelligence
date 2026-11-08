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
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e69500;">
            <div class="metric-title">Put-Backs (Returned)</div>
            <div class="metric-value">{total_putbacks}</div>
            <div class="sub-note">Kept Back on Shelf</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #28a745;">
            <div class="metric-title">Net Items Taken</div>
            <div class="metric-value">{net_taken}</div>
            <div class="sub-note">{'Carted / Bought' if net_taken > 0 else 'Zero (Returned)'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #6f42c1;">
            <div class="metric-title">Average Dwell Time</div>
            <div class="metric-value">{kpis['avg_dwell_seconds']}s</div>
            <div class="sub-note">Per Shopper Session</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Conversion Funnel & Zones",
        "🔥 Traffic & Dwell Heatmap",
        "👥 Customer Journey Logs",
        "🎥 Video Playback & HUD",
    ])

    # TAB 1: Conversion Funnel & Zones
    with tab1:
        st.subheader("Aisle Conversion & Planogram Performance")
        fcol1, fcol2 = st.columns([1, 1])

        with fcol1:
            st.markdown("##### 🛒 Shopper Conversion Funnel")
            funnel_img_path = OUTPUT_DIR / "conversion_funnel.png"
            if funnel_img_path.exists():
                st.image(str(funnel_img_path), width="stretch")
            else:
                st.info("Funnel chart not generated yet.")

        with fcol2:
            st.markdown("##### 📍 Shelf Zone Breakdown")
            zone_data = []
            for z_name, z_info in kpis["zone_stats"].items():
                zone_data.append({
                    "Zone": z_info["name"],
                    "Shelf?": "Yes" if z_info["is_shelf"] else "Walkway",
                    "Visitors": z_info["unique_visitors"],
                    "Avg Dwell (s)": z_info["avg_dwell_sec"],
                    "Total Dwell (s)": z_info["total_dwell_sec"],
                    "Interactions (Picks)": z_info["interactions"],
                })
            df_zones = pd.DataFrame(zone_data)
            st.dataframe(df_zones, width="stretch", hide_index=True)

            # Zone Dwell comparison bar chart
            st.bar_chart(df_zones.set_index("Zone")["Avg Dwell (s)"], height=230)

    # TAB 2: Traffic & Dwell Heatmap
    with tab2:
        st.subheader("Store Traffic & Dwell-Time Heatmap")
        st.write("Overhead spatial density representation showing hot zones where shoppers linger most:")
        heatmap_img_path = OUTPUT_DIR / "aisle_heatmap.png"
        if heatmap_img_path.exists():
            st.image(str(heatmap_img_path), caption="Aisle Density Heatmap (Red = High Dwell, Blue = Low)", width="stretch")
        else:
            st.info("Heatmap not found.")

    # TAB 3: Individual Customer Journeys
    with tab3:
        st.subheader("Shopper Session Log & Behavioral States")
        scol1, scol2 = st.columns(2)
        with scol1:
            selected_state = st.selectbox(
                "Filter by Highest State:",
                ["ALL", "INTERACTING", "BROWSING", "PASSING"],
            )
        with scol2:
            shelves = ["ALL"] + [z for z in sessions["primary_shelf"].unique() if z != "None"]
            selected_shelf = st.selectbox("Filter by Primary Shelf:", shelves)

        filtered_df = sessions.copy()
        if selected_state != "ALL":
            filtered_df = filtered_df[filtered_df["highest_state"] == selected_state]
        if selected_shelf != "ALL":
            filtered_df = filtered_df[filtered_df["primary_shelf"] == selected_shelf]

        st.dataframe(filtered_df, width="stretch", hide_index=True)

    # TAB 4: Video Playback
    with tab4:
        st.subheader("Annotated Video Stream & HUD Overlay")
        mp4_files = sorted(list(OUTPUT_DIR.glob("*.mp4")))
        if mp4_files:
            file_options = {p.name: p for p in mp4_files}
            default_key = "annotated_retail_analysis.mp4" if "annotated_retail_analysis.mp4" in file_options else list(file_options.keys())[0]
            selected_file = st.selectbox("Select Annotated Video to Watch:", list(file_options.keys()), index=list(file_options.keys()).index(default_key))
            video_out_path = file_options[selected_file]
            st.video(str(video_out_path))
            st.caption(f"📁 Video Location: `{video_out_path}`")
        else:
            st.info("Annotated video not found. Run `python main.py` with `--save-video` to generate it.")


if __name__ == "__main__":
    main()
