"""
Career quality scoring.

EDA Finding 6: Education fields are synthetic placeholders
("Tier-3 Engineering College", "Local Engineering College").
The 17.5% PhD rate is unrealistic for a real talent pool.
Education signals add noise, not signal — removed entirely.

Weights redistributed:
  product_company_score: 0.45 (unchanged from spec)
  progression_score:     0.40 (increased from 0.35 — absorbed education weight)
  tenure_score:          0.15 (decreased from 0.20)
"""

import re
from src.config import CONSULTING_FIRM_PATTERNS, EXP_OPTIMAL_YEARS
from src.utils.normaliser import experience_band_score


def compute_career_quality_score(candidate):
    """
    Three sub-scores (no education component — EDA Finding 6):

    1. product_company_score (weight 0.50):
       Ratio of roles at product companies vs. services/consulting companies.
       Product companies: any company NOT matching CONSULTING_FIRM_PATTERNS.
       Score = product_roles / total_roles.

    2. progression_score (weight 0.35):
       Detect upward title progression across career.
       Score 1.0 if clear seniority increase across at least 3 roles.
       Score 0.6 if lateral movement only.
       Score 0.2 if apparent downward movement.
       Detect by checking for Junior -> Mid -> Senior -> Lead -> Principal pattern.

    3. tenure_score (weight 0.15):
       Mean role duration across all roles.
       Optimal: 18-36 months per role.
       < 6 months average: 0.2 (job hopper signal)
       6-12 months: 0.5
       12-36 months: 1.0
       > 48 months: 0.7 (too slow-moving for founding team)

     career_quality_score = 0.45*product + 0.40*progression + 0.15*tenure
    """
    career = candidate.career
    if not career:
        return 0.0

    n_roles = len(career)

    # ── 1. product_company_score ──
    product_roles = 0
    for role in career:
        company = (role.company or "").lower().strip()
        is_consulting = False
        for pat in CONSULTING_FIRM_PATTERNS:
            if pat in company:
                is_consulting = True
                break
        if not is_consulting:
            product_roles += 1

    product_score = product_roles / n_roles if n_roles > 0 else 0.0

    # ── 2. progression_score ──
    seniority_levels = [
        "junior", "associate", "intern", "trainee",
        "mid", "senior", "lead", "principal", "staff",
        "director", "head", "vp", "chief", "cto",
    ]

    def _seniority_rank(title):
        t = (title or "").lower()
        # Check from most senior to least
        for i, level in enumerate(seniority_levels):
            if level in t:
                return i
        return -1  # neutral if no keyword found

    ranks = []
    for role in career:
        ranks.append(_seniority_rank(role.title))

    if len(ranks) >= 3:
        # Check for clear upward progression
        increases = sum(1 for i in range(1, len(ranks)) if ranks[i] > ranks[i - 1])
        decreases = sum(1 for i in range(1, len(ranks)) if ranks[i] < ranks[i - 1])
        if increases >= 2 and decreases == 0:
            progression_score = 1.0
        elif increases >= decreases:
            progression_score = 0.6
        else:
            progression_score = 0.2
    else:
        progression_score = 0.5  # neutral for too few roles

    # ── 3. tenure_score ──
    total_dur = sum(getattr(role, "duration_months", 0) or 0 for role in career)
    mean_dur = total_dur / n_roles if n_roles > 0 else 0

    if mean_dur < 6:
        tenure_score = 0.2
    elif mean_dur < 12:
        tenure_score = 0.5
    elif mean_dur <= 36:
        tenure_score = 1.0
    elif mean_dur <= 48:
        tenure_score = 0.8
    else:
        tenure_score = 0.7

    # ── combine ──
    score = 0.45 * product_score + 0.40 * progression_score + 0.15 * tenure_score

    # ── experience band multiplier ──
    # Penalises candidates with too little or too much experience.
    # Asymmetric: under-experience penalised harder than over-experience.
    exp_band = experience_band_score(candidate.experience_years)
    score = score * exp_band

    return max(0.0, min(1.0, score))
