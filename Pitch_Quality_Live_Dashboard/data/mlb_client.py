
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests

BASE = "https://statsapi.mlb.com"
SCHEDULE_URL = f"{BASE}/api/v1/schedule"
LIVE_FEED_URL = f"{BASE}/api/v1.1/game/{{game_pk}}/feed/live"

REQUEST_TIMEOUT = 10  # seconds


@dataclass
class Pitch:
    game_pk: int
    play_id: str            # unique id for this pitch event
    at_bat_index: int
    pitch_number: int
    inning: int
    half_inning: str        # "top" / "bottom"
    pitcher: str
    batter: str
    pitch_type: Optional[str]        # e.g. "FF", "SL", "CH"
    pitch_type_desc: Optional[str]   # e.g. "Four-Seam Fastball"
    velocity: Optional[float]        # mph, start speed
    spin_rate: Optional[float]       # rpm
    plate_x: Optional[float]         # horizontal location (ft, catcher's view)
    plate_z: Optional[float]         # vertical location (ft)
    sz_top: Optional[float]          # top of batter's strike zone (ft)
    sz_bot: Optional[float]          # bottom of batter's strike zone (ft)
    result: Optional[str]            # e.g. "Ball", "Called Strike", "In play, out(s)"
    timestamp: str


def get_todays_games() -> list[dict]:
    """Return today's MLB games with basic status info.

    Each dict: {game_pk, away, home, status, start_time}
    """
    params = {"sportId": 1}
    resp = requests.get(SCHEDULE_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    games = []
    for date_entry in payload.get("dates", []):
        for g in date_entry.get("games", []):
            games.append(
                {
                    "game_pk": g["gamePk"],
                    "away": g["teams"]["away"]["team"]["name"],
                    "home": g["teams"]["home"]["team"]["name"],
                    "status": g["status"]["detailedState"],
                    "start_time": g.get("gameDate"),
                }
            )
    return games


def _extract_pitches_from_play(game_pk: int, play: dict) -> list[Pitch]:
    """Pull every pitch event out of a single 'play' (at-bat) block."""
    pitches: list[Pitch] = []

    matchup = play.get("matchup", {})
    pitcher_name = matchup.get("pitcher", {}).get("fullName", "Unknown")
    batter_name = matchup.get("batter", {}).get("fullName", "Unknown")
    about = play.get("about", {})
    at_bat_index = about.get("atBatIndex", -1)
    inning = about.get("inning")
    half = "top" if about.get("isTopInning") else "bottom"

    for event in play.get("playEvents", []):
        if event.get("isPitch") is not True:
            continue

        pitch_data = event.get("pitchData", {}) or {}
        details = event.get("details", {}) or {}
        coords = pitch_data.get("coordinates", {}) or {}
        breaks = pitch_data.get("breaks", {}) or {}

        play_id = event.get("playId") or f"{game_pk}-{at_bat_index}-{event.get('pitchNumber')}"

        pitches.append(
            Pitch(
                game_pk=game_pk,
                play_id=play_id,
                at_bat_index=at_bat_index,
                pitch_number=event.get("pitchNumber", 0),
                inning=inning,
                half_inning=half,
                pitcher=pitcher_name,
                batter=batter_name,
                pitch_type=details.get("type", {}).get("code"),
                pitch_type_desc=details.get("type", {}).get("description"),
                velocity=pitch_data.get("startSpeed"),
                spin_rate=breaks.get("spinRate"),
                plate_x=coords.get("pX"),
                plate_z=coords.get("pZ"),
                sz_top=pitch_data.get("strikeZoneTop"),
                sz_bot=pitch_data.get("strikeZoneBottom"),
                result=details.get("description"),
                timestamp=event.get("startTime", ""),
            )
        )

    return pitches


def fetch_all_pitches(game_pk: int) -> list[Pitch]:
    """Fetch the full live feed and return every pitch thrown so far, in order."""
    url = LIVE_FEED_URL.format(game_pk=game_pk)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    all_plays = payload.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches: list[Pitch] = []
    for play in all_plays:
        pitches.extend(_extract_pitches_from_play(game_pk, play))
    return pitches


class LivePitchPoller:
    """Stateful poller: call `.poll()` repeatedly to get only NEW pitches
    since the last call, for a given game.
    """

    def __init__(self, game_pk: int):
        self.game_pk = game_pk
        self._seen_play_ids: set[str] = set()

    def poll(self) -> list[Pitch]:
        pitches = fetch_all_pitches(self.game_pk)
        new_pitches = [p for p in pitches if p.play_id not in self._seen_play_ids]
        for p in new_pitches:
            self._seen_play_ids.add(p.play_id)
        return new_pitches


if __name__ == "__main__":
    # Quick manual smoke test: list today's games.
    for g in get_todays_games():
        print(g["game_pk"], g["away"], "@", g["home"], "-", g["status"])
