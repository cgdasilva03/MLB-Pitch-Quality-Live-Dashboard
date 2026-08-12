"""
quality.py

A transparent "Stuff Score" for individual pitches, built as a stand-in for
proprietary metrics like Stuff+ (which require private outcome-modeling data
we don't have access to).

Approach:
  For each pitch type, compare velocity and spin rate to a league-average
  baseline for that pitch type. Pitches that are faster and spin more than
  average for their type score higher. Location is scored separately using
  distance from the edge of the strike zone (edge pitches > down-the-middle
  or way-out-of-zone pitches, since elite command lives on the edges).

  Final score is 20-80 scouting-scale style (matches how MLB itself grades
  tools), with 50 = league average.

These league averages are approximate 2023-2024 Statcast full-season
figures. They're a reasonable baseline, not gospel -- swap in real
season-to-date averages (pulled via pybaseball) for more accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

# Approximate league-average velocity (mph) and spin rate (rpm) by pitch type code.
# Source: rough Statcast full-season averages, MLB-wide.
LEAGUE_AVERAGES: dict[str, dict[str, float]] = {
    "FF": {"velocity": 94.0, "spin": 2300},  # four-seam fastball
    "SI": {"velocity": 93.5, "spin": 2150},  # sinker
    "FC": {"velocity": 89.0, "spin": 2400},  # cutter
    "SL": {"velocity": 85.0, "spin": 2450},  # slider
    "ST": {"velocity": 82.5, "spin": 2500},  # sweeper
    "CU": {"velocity": 79.0, "spin": 2600},  # curveball
    "KC": {"velocity": 80.5, "spin": 2650},  # knuckle curve
    "CH": {"velocity": 85.5, "spin": 1750},  # changeup
    "FS": {"velocity": 86.5, "spin": 1400},  # splitter
    "KN": {"velocity": 68.0, "spin": 1200},  # knuckleball
}

DEFAULT_AVG = {"velocity": 88.0, "spin": 2200}


@dataclass
class QualityBreakdown:
    velocity_score: float   # 20-80 scale
    spin_score: float       # 20-80 scale
    location_score: float   # 20-80 scale
    overall_score: float    # 20-80 scale, weighted blend
    grade: str               # human label


def _to_scouting_scale(value: float, avg: float, spread: float) -> float:
    """Convert a raw stat into a 20-80 scouting-scale score.

    50 = average. Each `spread` unit above/below average moves the score
    10 points, clamped to [20, 80]. This mirrors how MLB scouts grade tools.
    """
    z = (value - avg) / spread if spread else 0
    score = 50 + (z * 10)
    return max(20.0, min(80.0, score))


def _location_score(plate_x: float | None, plate_z: float | None,
                     sz_top: float | None, sz_bot: float | None) -> float:
    """Score location based on proximity to the strike zone edges.

    Pitches right on the black (edge of the zone) score highest -- that's
    where called strikes and weak contact live. Pitches middle-middle score
    respectably (it's still a strike) but lower than the corners. Pitches
    outside the zone are scored on how far outside they are -- a pitch just
    off the black is very different from one bounced in the dirt, and this
    treats them that way instead of collapsing both toward "average."
    """
    if plate_x is None or plate_z is None or sz_top is None or sz_bot is None:
        return 50.0  # neutral/unknown

    # Strike zone horizontal half-width is ~0.83 ft (17in plate / 2 + ball radius).
    half_width = 0.83

    # How far outside the zone on each axis (0 if inside on that axis).
    outside_x = max(0.0, abs(plate_x) - half_width)
    if plate_z > sz_top:
        outside_z = plate_z - sz_top
    elif plate_z < sz_bot:
        outside_z = sz_bot - plate_z
    else:
        outside_z = 0.0

    if outside_x > 0 or outside_z > 0:
        # Outside the zone: score drops the further out it is (real distance,
        # not just the smaller of the two axes).
        outside_dist = (outside_x ** 2 + outside_z ** 2) ** 0.5
        score = 50 - (outside_dist * 35)
        return max(20.0, min(50.0, score))

    # Inside the zone (it's a strike): reward proximity to the nearest edge
    # over the middle of the plate, since edges generate more called strikes
    # and weak contact than pitches grooved down the middle.
    margin_x = half_width - abs(plate_x)
    margin_z = min(plate_z - sz_bot, sz_top - plate_z)
    margin = min(margin_x, margin_z)
    score = 72 - (margin * 22)
    return max(50.0, min(80.0, score))


def score_pitch(pitch_type: str | None, velocity: float | None,
                 spin_rate: float | None, plate_x: float | None,
                 plate_z: float | None, sz_top: float | None,
                 sz_bot: float | None) -> QualityBreakdown:
    """Compute a 20-80 quality breakdown for a single pitch."""
    avg = LEAGUE_AVERAGES.get(pitch_type, DEFAULT_AVG) if pitch_type else DEFAULT_AVG

    velocity_score = (
        _to_scouting_scale(velocity, avg["velocity"], spread=2.5)
        if velocity is not None else 50.0
    )
    spin_score = (
        _to_scouting_scale(spin_rate, avg["spin"], spread=250)
        if spin_rate is not None else 50.0
    )
    location_score = _location_score(plate_x, plate_z, sz_top, sz_bot)

    # Weighted blend: velocity/spin ("stuff") matter, but location ("command")
    # is at least as predictive of outcomes, so weight it heavily.
    overall = (velocity_score * 0.3) + (spin_score * 0.3) + (location_score * 0.4)

    if overall >= 70:
        grade = "Elite"
    elif overall >= 60:
        grade = "Plus"
    elif overall >= 45:
        grade = "Average"
    elif overall >= 35:
        grade = "Below Average"
    else:
        grade = "Poor"

    return QualityBreakdown(
        velocity_score=round(velocity_score, 1),
        spin_score=round(spin_score, 1),
        location_score=round(location_score, 1),
        overall_score=round(overall, 1),
        grade=grade,
    )
