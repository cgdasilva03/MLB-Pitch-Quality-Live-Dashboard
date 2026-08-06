"""
dashboard.py

Streamlit dashboard for live MLB pitch tracking.

Run with:
    streamlit run app/dashboard.py

What it does:
  1. Lets you pick a live/in-progress game from today's schedule.
  2. Polls the MLB live feed on a timer.
  3. Plots every pitch's location in the strike zone, colored by pitch type.
  4. Shows a live table of recent pitches with velocity, spin, and a
     20-80 "Stuff Score" quality grade.
  5. Tracks rolling per-pitcher stats: velocity trend, pitch mix, avg quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Allow importing sibling `data/` package regardless of cwd.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.mlb_client import LivePitchPoller, get_todays_games  # noqa: E402
from data.quality import score_pitch  # noqa: E402

st.set_page_config(page_title="Live Pitch Tracker", layout="wide")

PITCH_TYPE_COLORS = {
    "FF": "#e63946", "SI": "#f4a261", "FC": "#e9c46a", "SL": "#2a9d8f",
    "ST": "#264653", "CU": "#457b9d", "KC": "#1d3557", "CH": "#8ac926",
    "FS": "#6a4c93", "KN": "#adb5bd",
}
DEFAULT_COLOR = "#888888"


def init_state():
    if "pitches" not in st.session_state:
        st.session_state.pitches = []  # list of dicts, most recent last
    if "poller" not in st.session_state:
        st.session_state.poller = None
    if "game_pk" not in st.session_state:
        st.session_state.game_pk = None


def pitch_to_row(pitch, quality) -> dict:
    return {
        "at_bat_index": pitch.at_bat_index,
        "inning": pitch.inning,
        "half": pitch.half_inning,
        "pitcher": pitch.pitcher,
        "batter": pitch.batter,
        "pitch_type": pitch.pitch_type or "?",
        "pitch_desc": pitch.pitch_type_desc or "Unknown",
        "velocity": pitch.velocity,
        "spin_rate": pitch.spin_rate,
        "plate_x": pitch.plate_x,
        "plate_z": pitch.plate_z,
        "sz_top": pitch.sz_top,
        "sz_bot": pitch.sz_bot,
        "result": pitch.result,
        "quality_score": quality.overall_score,
        "grade": quality.grade,
    }


def render_strike_zone(df: pd.DataFrame):
    fig = go.Figure()

    # Draw an average strike zone box (using median top/bottom from the data,
    # falling back to standard MLB zone if not enough data yet).
    sz_top = df["sz_top"].dropna().median() if not df["sz_top"].dropna().empty else 3.5
    sz_bot = df["sz_bot"].dropna().median() if not df["sz_bot"].dropna().empty else 1.5

    fig.add_shape(
        type="rect", x0=-0.83, x1=0.83, y0=sz_bot, y1=sz_top,
        line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0.05)",
    )

    for ptype, group in df.groupby("pitch_type"):
        fig.add_trace(
            go.Scatter(
                x=group["plate_x"], y=group["plate_z"],
                mode="markers",
                name=ptype,
                marker=dict(
                    size=14,
                    color=PITCH_TYPE_COLORS.get(ptype, DEFAULT_COLOR),
                    line=dict(width=1, color="white"),
                ),
                text=group.apply(
                    lambda r: f"{r['pitch_desc']} — {r['velocity']} mph — {r['grade']}",
                    axis=1,
                ),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        xaxis=dict(range=[-2.5, 2.5], title="Horizontal (ft, catcher's view)", zeroline=False),
        yaxis=dict(range=[0, 5], title="Height (ft)", zeroline=False),
        height=550,
        plot_bgcolor="#111",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend_title="Pitch Type",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    init_state()
    st.title("⚾ Live Pitch Tracker")
    st.caption("Real-time pitch location, type, velocity, and quality grading from MLB's live feed.")

    with st.sidebar:
        st.header("Game Selection")
        try:
            games = get_todays_games()
        except Exception as e:
            st.error(f"Couldn't load today's schedule: {e}")
            games = []

        if not games:
            st.info("No games found for today.")
            return

        options = {
            f"{g['away']} @ {g['home']} ({g['status']})": g["game_pk"] for g in games
        }
        choice = st.selectbox("Choose a game", list(options.keys()))
        selected_pk = options[choice]

        poll_interval = st.slider("Poll interval (seconds)", 2, 15, 5)
        auto = st.checkbox("Auto-refresh", value=True)

        if st.button("Start / switch to this game") or st.session_state.game_pk is None:
            if st.session_state.game_pk != selected_pk:
                st.session_state.game_pk = selected_pk
                st.session_state.poller = LivePitchPoller(selected_pk)
                st.session_state.pitches = []

    if st.session_state.poller is None:
        st.info("Select a game and click **Start / switch to this game**.")
        return

    if auto:
        st_autorefresh(interval=poll_interval * 1000, key="refresh")

    # Poll for new pitches and append.
    try:
        new_pitches = st.session_state.poller.poll()
        for p in new_pitches:
            q = score_pitch(p.pitch_type, p.velocity, p.spin_rate,
                             p.plate_x, p.plate_z, p.sz_top, p.sz_bot)
            st.session_state.pitches.append(pitch_to_row(p, q))
    except Exception as e:
        st.warning(f"Poll failed (will retry): {e}")

    if not st.session_state.pitches:
        st.info("Waiting for the first pitch of this game...")
        return

    df = pd.DataFrame(st.session_state.pitches)

    with st.sidebar:
        st.header("Filters")
        pitcher_options = ["All pitchers"] + sorted(df["pitcher"].unique().tolist())
        selected_pitcher = st.selectbox("Pitcher", pitcher_options)
        current_at_bat_only = st.checkbox("Current at-bat only", value=False)

    filtered = df
    if selected_pitcher != "All pitchers":
        filtered = filtered[filtered["pitcher"] == selected_pitcher]
    if current_at_bat_only:
        latest_at_bat = df["at_bat_index"].max()
        filtered = filtered[filtered["at_bat_index"] == latest_at_bat]

    if filtered.empty:
        st.info("No pitches match the current filters yet.")
        return

    filter_desc_bits = []
    if selected_pitcher != "All pitchers":
        filter_desc_bits.append(selected_pitcher)
    if current_at_bat_only:
        filter_desc_bits.append("current at-bat")
    filter_desc = f" ({', '.join(filter_desc_bits)})" if filter_desc_bits else ""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Strike Zone{filter_desc}")
        render_strike_zone(filtered)

    with col2:
        st.subheader("Latest Pitches")
        recent = filtered.tail(15)[
            ["pitcher", "pitch_desc", "velocity", "spin_rate", "grade", "result"]
        ].iloc[::-1]
        st.dataframe(recent, use_container_width=True, hide_index=True)

    st.subheader(f"Pitcher Summary{filter_desc}")
    summary = (
        filtered.groupby("pitcher")
        .agg(
            pitches=("pitch_type", "count"),
            avg_velocity=("velocity", "mean"),
            avg_spin=("spin_rate", "mean"),
            avg_quality=("quality_score", "mean"),
        )
        .round(1)
        .sort_values("pitches", ascending=False)
    )
    st.dataframe(summary, use_container_width=True)

    st.subheader(f"Pitch Mix{filter_desc}")
    mix = filtered["pitch_desc"].value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(mix)


if __name__ == "__main__":
    main()
