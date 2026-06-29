"""
Behavioural reliability scoring.

NOTE — Double-counting avoidance (from config.CORRELATED_SIGNAL_PAIRS):
The EDA found that recruiter_response_rate, is_active, interview_completion_rate,
notice_period_days, and open_to_work_flag are all uncorrelated (max |r| = 0.036).
This means none of these signals overlaps significantly with another. The
behavioural score can safely include availability (open_to_work, is_active,
notice_period), reliability (response_rate, interview_completion, ghosting),
and engagement (remaining numeric signals) as independent sub-components
without risking double-counting.

NOTE — Sentinel-1 handling (from EDA Section 5):
github_activity_score has median = -1.00, mean = 9.62, max = 96.9 — majority are -1
  sentinel → 0.4 (neutral), 0 → 0.1, else → min(val/96.9, 1.0)
offer_acceptance_rate has median = -1.00, mean = -0.40 — same pattern
  sentinel → 0.4 (neutral), else → use raw value (already 0-1)
-1 is a sentinel meaning "no data" (not connected / no history).
These must NOT be treated as negative signals.

NOTE — EDA-driven adjustments:
notice_period_days: mean=87, median=90, max=150 → denominator widened to 120
  (30d → 1.0, 90d → 0.5, 150d → 0.0)
avg_response_time_hours: mean=132.7, max=280 → formula: max(0, 1 - (h-2)/280)
  (2h → 1.0, 140h → 0.5, 280h → 0.0)
"""


def normalise_sentinel(value, default=0.3):
    """Return default if value == -1 (sentinel), else return value unchanged."""
    if value == -1.0 or value == -1:
        return default
    return value


def compute_behavioural_score(candidate, calibration):
    """
    Composite of RedRob signals. All signals first normalised using calibration bounds.

    Sub-scores (each 0.0-1.0):

    1. availability_score (weight 0.35):
       - open_to_work: 1.0 if True, 0.4 if False
       - is_active: multiply by 1.0 if True, 0.6 if False
       - notice_period_days: score = max(0, 1.0 - (days - 30) / 60)
         (30 days or less = 1.0, 90 days = 0.0)

    2. reliability_score (weight 0.40):
       - response_rate: use raw value (already 0.0-1.0)
       - interview_completion_rate: use raw value
       - ghosting penalty: max(0, 1.0 - ghosting_count * 0.2)
       reliability_score = mean of these three

    3. engagement_score (weight 0.25):
       From remaining raw signals in behavioural.raw dict.
       Compute mean of all numeric signal values after normalising each to [0,1]
       using calibration['behavioural_signal_ranges'][signal_name].
       Ignore boolean signals in this sub-score.

    behavioural_score = 0.35*availability + 0.40*reliability + 0.25*engagement
    Clamp to [0.0, 1.0].
    """
    signals = candidate.behavioural
    raw = signals.raw

    # ── sub-score 1: availability ──
    open_score = 1.0 if signals.open_to_work else 0.4
    active_mult = 1.0 if signals.is_active else 0.6

    notice = signals.notice_period_days
    # EDA: mean=87, median=90, max=150 — adjusted denominator from 60 to 120
    notice_score = max(0.0, 1.0 - (notice - 30) / 120)

    availability = 0.35 * open_score + 0.35 * active_mult + 0.30 * notice_score

    # ── sub-score 2: reliability ──
    response = signals.response_rate
    interview = signals.interview_completion_rate
    ghosting = signals.ghosting_count
    ghost_penalty = max(0.0, 1.0 - ghosting * 0.2)

    reliability = (response + interview + ghost_penalty) / 3.0

    # ── sub-score 3: engagement ──
    # Numeric signals from raw dict, excluding those used above
    used_signals = {
        "recruiter_response_rate", "interview_completion_rate",
        "notice_period_days", "open_to_work_flag",
    }
    engagement_vals = []
    for sig_name, sig_value in raw.items():
        if sig_name in used_signals:
            continue
        if not isinstance(sig_value, (int, float)):
            continue
        if isinstance(sig_value, bool):
            continue
        # Handle sentinel -1
        sig_value = normalise_sentinel(sig_value)
        # Normalise using calibration ranges if available
        ranges = calibration.get("behavioural_signal_ranges", {})
        if sig_name in ranges:
            lo = ranges[sig_name]["min"]
            hi = ranges[sig_name]["max"]
            if hi > lo:
                normalised = (sig_value - lo) / (hi - lo)
                engagement_vals.append(max(0.0, min(1.0, normalised)))
            else:
                engagement_vals.append(0.5)
        else:
            # Fallback: clamp known ranges
            if sig_name == "github_activity_score":
                # EDA: median=-1 (sentinel), max=96.9
                if sig_value is None or sig_value == -1 or sig_value == -1.0:
                    engagement_vals.append(0.4)  # neutral — no data
                elif sig_value == 0:
                    engagement_vals.append(0.1)  # active but zero activity
                else:
                    engagement_vals.append(min(float(sig_value) / 96.9, 1.0))
            elif sig_name == "offer_acceptance_rate":
                # EDA: median=-1 (sentinel), range 0-1 when present
                if sig_value is None or sig_value == -1 or sig_value == -1.0:
                    engagement_vals.append(0.4)  # neutral — no offers made
                else:
                    engagement_vals.append(float(sig_value))  # already 0-1
            elif sig_name == "avg_response_time_hours":
                # EDA: mean=132.7, max=280
                val = float(sig_value)
                score = max(0.0, 1.0 - (val - 2) / 280.0)
                engagement_vals.append(score)
            elif sig_name == "profile_completeness_score":
                engagement_vals.append(sig_value / 100.0)
            else:
                engagement_vals.append(0.5)

    engagement = sum(engagement_vals) / len(engagement_vals) if engagement_vals else 0.5

    # ── combine ──
    score = 0.35 * availability + 0.40 * reliability + 0.25 * engagement
    return max(0.0, min(1.0, score))
