"""
Phase 1 — Broad Filter.

Four-gate design from build_spec.md:
- Gate 1: Honeypot elimination
- Gate 2: Behavioural hard gate
- Gate 3: Role archetype filter (tiered archetype scoring + consulting)
- Gate 4: Experience band floor/ceiling

Gate 3 redesigned per EDA finding (only 0.9% have AI titles).
Four-tier scoring: hard exclude, strong signal, weak signal, ambiguous.
Never hard-exclude a candidate whose career history contains ML/AI work.
"""

from src.config import (
    MIN_RESPONSE_RATE, MIN_INTERVIEW_COMPLETION, MAX_GHOSTING_COUNT,
    MIN_EXPERIENCE_YEARS, MAX_EXPERIENCE_YEARS, MIN_ARCHETYPE_SCORE,
    HARD_EXCLUDE_TITLES, AI_ARCHETYPE_TITLES, CONSULTING_FIRM_PATTERNS,
)
from src.utils.honeypot import (
    check_chronological_consistency,
    check_skill_duration_sanity,
    compute_honeypot_score,
)

import re

TIER1_DESC_KEYWORDS = [
    "engineer", "developer", "scientist",
    "analyst", "architect", "researcher",
    "data", "python", "model", "algorithm",
]

# Word boundary patterns to prevent "ml" matching inside "html"
# and "ai" matching inside "email", "maintain", etc.
_ML_SHORT_PATTERN = re.compile(r'\bml\b', re.IGNORECASE)
_AI_SHORT_PATTERN = re.compile(r'\bai\b', re.IGNORECASE)
_ML_AI_LONG_PATTERN = re.compile(
    r'\b(machine learning|deep learning|nlp|llm|computer vision|'
    r'neural network|transformer|artificial intelligence)\b',
    re.IGNORECASE
)

TECHNICAL_TITLES = [
    "software engineer", "backend", "full stack", "cloud", "devops",
]

AMBIGUOUS_TITLES = [
    "analytics engineer", "data analyst", "data engineer",
]


def _has_ml_ai_in_career(candidate):
    """Check if any role with a technical or AI title has ML/AI keywords.
    
    Only counts ML/AI mentions in roles whose title is technical or AI.
    This prevents false positives from "AI-strategy advisory" appearing in
    the descriptions of non-technical roles (Customer Support, Accountant,
    etc.) — a known synthetic data artifact.
    """
    tech_titles_set = set(AI_ARCHETYPE_TITLES + TECHNICAL_TITLES)
    for role in candidate.career:
        title = (role.title or "").lower()
        # Must have a technical/AI title for the ML/AI mention to count
        is_tech_title = any(pat in title for pat in tech_titles_set)
        if not is_tech_title:
            continue
        text = (role.title or "") + " " + (role.description or "")
        if _ML_SHORT_PATTERN.search(text):
            return True
        if _AI_SHORT_PATTERN.search(text):
            return True
        if _ML_AI_LONG_PATTERN.search(text):
            return True
    return False


def _role_has_technical_desc(role):
    """Check if a single role has technical keywords in its description."""
    desc = (role.description or "").lower()
    title = (role.title or "").lower()
    combined = f"{desc} {title}"
    for kw in TIER1_DESC_KEYWORDS:
        if kw in combined:
            return True
    return False


def _has_any_technical_desc(candidate):
    """Check if any role in career history has a technical description."""
    for role in candidate.career:
        if _role_has_technical_desc(role):
            return True
    return False


def _current_title(candidate):
    """Get current (most recent) role title, lowercased."""
    if candidate.career:
        return (candidate.career[0].title or "").lower()
    profile = getattr(candidate, "profile", {}) if hasattr(candidate, "profile") else {}
    return (profile.get("current_title") or "").strip().lower()


def _title_matches_list(title, patterns):
    """Check if title contains any of the given patterns."""
    for pat in patterns:
        if pat in title:
            return True
    return False


def is_confirmed_non_ai(candidate):
    """
    True only if current title is in HARD_EXCLUDE_TITLES
    AND zero roles have technical descriptions.
    A 'Marketing Manager' who built ML pipelines is not excluded.
    """
    current = _current_title(candidate)
    if not current:
        return False
    matched = _title_matches_list(current, HARD_EXCLUDE_TITLES)
    if not matched:
        return False
    if _has_any_technical_desc(candidate):
        return False
    return True


def is_consulting_only(candidate):
    """
    True only if EVERY role is at a consulting/services firm.
    A single product-company role redeems the candidate.
    """
    if not candidate.career:
        return False
    for role in candidate.career:
        company = (role.company or "").lower().strip()
        matched = any(pat in company for pat in CONSULTING_FIRM_PATTERNS)
        if not matched:
            return False
    return True


def _has_technical_title_in_career(candidate):
    """Check if ANY role has a title matching AI_ARCHETYPE_TITLES or TECHNICAL_TITLES."""
    for role in candidate.career:
        title = (role.title or "").lower()
        if _title_matches_list(title, AI_ARCHETYPE_TITLES):
            return True
        if _title_matches_list(title, TECHNICAL_TITLES):
            return True
    return False


def archetype_score(candidate):
    """
    Four-tier scoring per spec. Always returns a float.

    Rule: Never hard-exclude a candidate whose career history contains
    ML/AI work, regardless of current title.

    Tier 1 — Hard exclude (return 0.0):
        Current title in HARD_EXCLUDE_TITLES AND zero roles with technical
        titles anywhere in career history. Bypassed if candidate has ML/AI
        work in career.

    Tier 2 — Strong signal (return 1.0 / 0.85):
        Title matches AI_ARCHETYPE_TITLES. 1.0 if current role, 0.85 if
        only a past role matches.

    Tier 3 — Weak signal (return 0.5):
        Technical title (software engineer, backend, full stack, cloud, devops)
        with AI/ML keywords in career history.

    Tier 4 — Ambiguous (return 0.3):
        Analytics engineer, data analyst, data engineer titles, but only if
        candidate also has ML/AI keywords in career history (otherwise 0.0).

    Default — 0.0 for no signal, except 0.3 if ML/AI work is present.
    """
    has_ml = _has_ml_ai_in_career(candidate)
    current = _current_title(candidate)

    # Tier 1 — Hard exclude (skipped if candidate has ML/AI work)
    if not has_ml:
        if _title_matches_list(current, HARD_EXCLUDE_TITLES):
            if not _has_technical_title_in_career(candidate):
                return 0.0

    # Tier 2 — AI archetype title in any role
    for i, role in enumerate(candidate.career):
        title = (role.title or "").lower()
        if _title_matches_list(title, AI_ARCHETYPE_TITLES):
            return 1.0 if i == 0 else 0.85

    # Tier 3 — Technical title + AI/ML keywords in career
    if has_ml:
        for role in candidate.career:
            title = (role.title or "").lower()
            if _title_matches_list(title, TECHNICAL_TITLES):
                return 0.5

    # Tier 4 — Ambiguous title (requires ML/AI work to get 0.3)
    if has_ml and _title_matches_list(current, AMBIGUOUS_TITLES):
        return 0.3

    # Default
    if has_ml:
        return 0.3
    return 0.0


def run_phase1(data_path, calibration):
    """
    Streams all candidates. Applies four sequential gates.
    Returns list of passing candidates.
    """
    from src.utils.loader import stream_candidates
    from src.config import OUTPUT_DIR
    import csv, os

    passed = []
    total_read = 0
    honeypot_excluded = 0
    gate1_excluded = 0
    gate2_excluded = 0
    gate3_excluded = 0
    gate4_excluded = 0

    for candidate in stream_candidates(data_path):
        total_read += 1

        # ── Gate 1: Honeypot elimination ──
        chrono_ok, chrono_reason = check_chronological_consistency(candidate)
        skill_ok, skill_reason = check_skill_duration_sanity(candidate)
        if not chrono_ok or not skill_ok:
            honeypot_excluded += 1
            continue

        honeypot_score = compute_honeypot_score(candidate)
        if honeypot_score >= 0.6:
            honeypot_excluded += 1
            continue

        # ── Gate 2: Behavioural hard gate ──
        bh = candidate.behavioural
        if bh.response_rate < MIN_RESPONSE_RATE and not bh.is_active:
            gate1_excluded += 1
            continue
        if bh.interview_completion_rate < MIN_INTERVIEW_COMPLETION and bh.ghosting_count > MAX_GHOSTING_COUNT:
            gate1_excluded += 1
            continue

        # ── Gate 3: Archetype + consulting filter ──
        if is_confirmed_non_ai(candidate):
            gate2_excluded += 1
            continue

        if is_consulting_only(candidate):
            gate2_excluded += 1
            continue

        a_score = archetype_score(candidate)
        if a_score < MIN_ARCHETYPE_SCORE:
            gate3_excluded += 1
            continue

        # ── Gate 4: Experience band ──
        exp = candidate.experience_years
        if exp < MIN_EXPERIENCE_YEARS or exp > MAX_EXPERIENCE_YEARS:
            gate4_excluded += 1
            continue

        passed.append(candidate)

    print(f"Total read: {total_read}")
    print(f"Honeypot excluded: {honeypot_excluded}")
    print(f"Gate 1 (behavioural) excluded: {gate1_excluded}")
    print(f"Gate 2 (non-AI + consulting) excluded: {gate2_excluded}")
    print(f"Gate 3 (archetype) excluded: {gate3_excluded}")
    print(f"Gate 4 (experience) excluded: {gate4_excluded}")
    print(f"Passing: {len(passed)}")

    # Write audit CSV
    audit_path = os.path.join(OUTPUT_DIR, "phase1_passed.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(audit_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "headline", "experience_years", "archetype_score"])
        for c in passed:
            writer.writerow([c.candidate_id, c.headline, c.experience_years, round(archetype_score(c), 4)])
    print(f"  Audit: {audit_path} ({len(passed)} rows)")

    return passed
