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
│   └── quality.py        # 20-80 heuristic "Stuff Score" pitch quality grading
├── ml/
│   ├── auto_train.py     # zero-argument autonomous trainer (dashboard calls this itself)
│   ├── train_model.py    # underlying training logic (feature engineering, LightGBM)
│   ├── model.py          # online: loads the trained model for live scoring, hot-reloads
│   └── models/           # trained model bundles land here (gitignored)
├── requirements.txt
└── README.md
```

## How pitch quality is scored — fully automatic

You never run a training command yourself. The dashboard manages the
whole lifecycle on its own:

1. On startup, it checks whether a trained model exists and how old it is.
2. If there's no model yet, or the existing one is more than 24 hours old,
   it **automatically launches training in the background** (a separate
   subprocess, so the dashboard itself never freezes or blocks).
3. While that's running, every pitch is graded with the heuristic scorer
   in the meantime — you always get a grade, never a blank spot.
4. Once training finishes, the dashboard picks up the new model file
   automatically on its next poll and starts using it — no restart, no
   button to click.
5. From then on, it re-checks freshness on every session and silently
   retrains itself again once the model passes 24 hours old.

The sidebar shows which state it's in ("training in background," "learned
model active," etc.) purely as a status readout — there's nothing to
configure there.

**The one real constraint**: training needs internet access to pull
Statcast data (via `pybaseball`), so it only works where the dashboard is
actually running has a connection. If it doesn't, or `pybaseball`/
`lightgbm` aren't installed, the dashboard just keeps using the heuristic
scorer indefinitely and shows why in the sidebar — it never crashes.

If you want the model to also refresh even when nobody has the dashboard
open (e.g. so it's ready fresh every game day without anyone launching the
app first), add a scheduled job that runs the same auto-trainer
independently:

```bash
# crontab -e
0 6 * * * cd /path/to/pitch-dashboard && python -m ml.auto_train
```

### Under the hood

- `ml/auto_train.py` — the zero-argument, self-scheduling trainer. Picks
  its own date range (current season's Opening Day through yesterday),
  uses a lock file so overlapping runs can't collide, and writes the
  model atomically so the dashboard never reads a half-written file.
- `ml/model.py` — loads the model for live inference and knows how to
  hot-reload itself when the file on disk changes.
- `ml/train_model.py` — the underlying training logic (feature
  engineering, target construction, LightGBM training) that both the
  manual and automatic entry points share.

Grading itself works the same way described before: a gradient-boosted
model predicts each pitch's run value using velocity, spin, movement,
location, count, and — critically — the batter's and pitcher's identity
as learned features, so it picks up on real matchup patterns rather than
applying one fixed formula to everyone.

## Extending this

- **Persistence**: pitches currently live only in the Streamlit session.
  Add a SQLite write in the poll loop if you want history across sessions.
- **Reuse with your betting agent panel**: this poller/client is a clean,
  separate module — the pitch-level data here (velocity trends, quality
  scores, pitch mix) could feed as a research signal into that project's
  analyst agent later on.
- **Model improvements**: swap in a neural embedding model instead of
  LightGBM categoricals if you want smoother generalization across
  batter/pitcher pairs with very little data; add park factors; add
  recent-N-pitch sequence features (e.g. "what did this pitcher just throw").

## Notes / limitations

- MLB's live feed updates are typically only seconds behind the actual
  pitch, not truly instantaneous.
- Some fields (especially spin rate) are occasionally missing/delayed in
  the raw feed — the code handles this gracefully by falling back to a
  neutral score rather than crashing.
- No API key required; this uses MLB's public Stats API.
