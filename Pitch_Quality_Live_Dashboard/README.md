# Live Pitch Tracker

A real-time dashboard for MLB pitch data — location, pitch type, velocity, and
a transparent 20-80 "Stuff Score" quality grade for every pitch, pulled
straight from MLB's live game feed as it happens.

## What it does

- Lists today's MLB games and lets you pick one that's live (or in progress)
- Polls MLB's public live feed every few seconds for new pitches
- Plots every pitch on a strike-zone chart, colored by pitch type
- Shows a live-updating table of recent pitches (velocity, spin rate, grade)
- Tracks rolling per-pitcher stats: avg velocity, avg spin, pitch mix, avg quality

## Setup

```bash
cd pitch-dashboard
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/dashboard.py
```

This opens a browser tab at `http://localhost:8501`. Pick a live game from
the sidebar and click **Start / switch to this game**. If no MLB games are
live right now, you'll still see the schedule — check back during game
hours (afternoons/evenings, US time, in season).

## Project structure

```
pitch-dashboard/
├── app/
│   └── dashboard.py      # Streamlit UI, polling loop, charts
├── data/
│   ├── mlb_client.py     # MLB Stats API client + live pitch poller
│   └── quality.py        # 20-80 "Stuff Score" pitch quality grading
├── requirements.txt
└── README.md
```

## How pitch quality is scored

There's no public "Stuff+" — that metric (and similar ones like PitchingBot)
is proprietary and trained on private outcome data. Instead, `quality.py`
builds a transparent proxy:

1. **Velocity** and **spin rate** are compared against league-average
   baselines *for that specific pitch type* (a 98mph fastball and a 98mph
   sinker aren't graded the same way a generic "fast pitch" would be).
2. **Location** is scored by distance from the edge of the strike zone —
   pitches on the black score highest, since that's where called strikes
   and weak contact cluster; middle-middle and well-out-of-zone pitches
   score lower.
3. These three combine into a 20-80 scouting-scale score (50 = average),
   the same scale scouts use for tools grades.

The league-average baselines in `quality.py` are approximate — swap in
real season-to-date averages (via `pybaseball`) if you want more precision.

## Extending this

- **Persistence**: pitches currently live only in the Streamlit session.
  Add a SQLite write in the poll loop if you want history across sessions.
- **Real Stuff+**: if you get access to historical Statcast + outcome data
  (pybaseball can pull this in bulk), you could train a real outcome-based
  model instead of the heuristic scorer.
- **Reuse with your betting agent panel**: this poller/client is a clean,
  separate module — the pitch-level data here (velocity trends, quality
  scores, pitch mix) could feed as a research signal into that project's
  analyst agent later on.

## Notes / limitations

- MLB's live feed updates are typically only seconds behind the actual
  pitch, not truly instantaneous.
- Some fields (especially spin rate) are occasionally missing/delayed in
  the raw feed — the code handles this gracefully by falling back to a
  neutral score rather than crashing.
- No API key required; this uses MLB's public Stats API.
