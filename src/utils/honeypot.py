"""
Honeypot detection — role-level consistency checks only.

EDA Finding 5: The ~80 "official" honeypots are the ~44 most egregious cases:
  - 19 candidates where sum(role.duration_months) > career_span + 24
  - 25 candidates where claimed experience > computable career span by 3+ years
The 17,887 candidates with skill duration > career span are LEGITIMATE —
skills naturally overlap across jobs (continuous learning).

Do NOT flag skill-duration sums. Only flag role-level inconsistencies.
Skill-duration checks in check_skill_duration_sanity() are soft-flag only
unless a single skill exceeds 2x the career span (hard exclude).
"""


def _parse_year(date_str):
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None


REFERENCE_DATE = 2026  # EDA max last_active_date year

def _get_career_span(candidate):
    """Compute career span in months from earliest start to latest end.
    For current roles (end_date = None), uses REFERENCE_DATE as the end."""
    earliest_start = None
    latest_end = None
    for role in candidate.career:
        raw = role.raw if hasattr(role, "raw") else {}
        start_str = raw.get("start_date", "")
        end_str = raw.get("end_date", "")
        start_y = _parse_year(start_str)
        if start_y and (earliest_start is None or start_y < earliest_start):
            earliest_start = start_y
        # Use REFERENCE_DATE for current roles
        if end_str:
            end_y = _parse_year(end_str)
        else:
            end_y = REFERENCE_DATE
        if end_y and (latest_end is None or end_y > latest_end):
            latest_end = end_y
    if earliest_start and latest_end and latest_end > earliest_start:
        return (latest_end - earliest_start) * 12 + 6
    if earliest_start and latest_end and latest_end == earliest_start:
        return 12  # minimum 1 year span
    return sum(r.duration_months for r in candidate.career)


def check_chronological_consistency(candidate):
    """
    Returns (is_clean, reason).

    Checks (role-level only):
    1. sum(role.duration_months) <= total_career_span_months + 24
       (allow 24-month overlap for job transitions — EDA: 19 fail this)
    2. claimed experience_years <= computable career span + 3 years
       (EDA: 25 fail this)
    3. Senior/Lead/Principal role starting before graduation_year - 2
       (strict: only the most implausible pre-grad cases)

    Returns (False, reason_string) if any check fails.
    Returns (True, "") if all pass.
    """
    career = candidate.career
    if not career:
        return True, ""

    career_span_months = _get_career_span(candidate)

    # Check 1: sum of role durations vs career span
    total_role_dur = sum(r.duration_months for r in career)
    if career_span_months > 0 and total_role_dur > career_span_months + 24:
        return False, (
            f"role durations ({total_role_dur}m) exceed career span "
            f"({career_span_months}m) by >24 months"
        )

    # Check 2: claimed experience vs actual career span
    exp_years = candidate.experience_years or 0
    exp_months = exp_years * 12
    if career_span_months > 0 and exp_months > career_span_months + 36:
        return False, (
            f"claimed experience ({exp_years:.1f}y) exceeds career span "
            f"({career_span_months / 12:.1f}y) by >3 years"
        )

    # Check 3: senior/lead role before graduation (only if egregious)
    education = candidate.education
    grad_year = None
    for edu in education:
        ey = edu.get("end_year") if isinstance(edu, dict) else None
        if ey and (grad_year is None or ey < grad_year):
            grad_year = ey

    if grad_year:
        for role in career:
            title = (role.title or "").lower()
            has_senior_title = any(
                kw in title for kw in ["senior", "lead", "principal"]
            )
            if has_senior_title:
                raw = role.raw if hasattr(role, "raw") else {}
                start_y = _parse_year(raw.get("start_date", ""))
                if start_y and start_y < grad_year - 2:
                    return False, (
                        f"senior role '{role.title}' started in {start_y}, "
                        f"which is >2 years before graduation year {grad_year}"
                    )

    return True, ""


def check_skill_duration_sanity(candidate):
    """
    Returns (is_clean, reason).

    Per EDA Finding 5: skill durations legitimately overlap across jobs.
    Do NOT flag sum of all skill durations — that's expected.

    Only check individual skills:
    - If a single skill's duration > career span AND endorsements == 0:
      soft flag — return (True, reason) to inform compute_honeypot_score
    - If a single skill's duration > 2x career span:
      hard exclude — return (False, reason)
    """
    career = candidate.career
    if not career:
        return True, ""

    career_span_months = _get_career_span(candidate)
    if career_span_months <= 0:
        return True, ""

    for skill in candidate.skills:
        dur = skill.duration_months or 0
        endorse = skill.endorsements or 0
        if dur > career_span_months * 2:
            return False, (
                f"skill '{skill.name}' has duration {dur}m exceeding "
                f"2x career span ({career_span_months}m)"
            )
        if dur > career_span_months and endorse == 0:
            return True, (
                f"skill '{skill.name}' has duration {dur}m exceeding "
                f"career span ({career_span_months}m) with zero endorsements"
            )

    return True, ""


def compute_honeypot_score(candidate):
    """
    Returns float 0.0 (clean) to 1.0 (almost certainly honeypot).

    Aggregates soft signals — does not hard-exclude, just informs gate:
    - Skills with 0 duration + 0 endorsements: +0.1 per skill, max 0.5
    - Experience years > career span by 3+ years: +0.3
    """
    score = 0.0

    zero_skills = 0
    for skill in candidate.skills:
        dur = skill.duration_months or 0
        endorse = skill.endorsements or 0
        if dur == 0 and endorse == 0:
            zero_skills += 1
    score += min(zero_skills * 0.1, 0.5)

    exp_years = candidate.experience_years or 0
    career_months = sum(r.duration_months for r in candidate.career)
    if career_months > 0 and exp_years * 12 > career_months + 36:
        score += 0.3

    return min(score, 1.0)
