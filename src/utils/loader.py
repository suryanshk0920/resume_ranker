import json
from src.models import Candidate, CareerRole, Skill, BehaviouralSignals

PROFICIENCY_MAP = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}


def stream_candidates(data_path):
    """
    Generator. Yields Candidate dataclass instances one at a time.
    Reads candidates.jsonl using jsonlines library.
    Performs field normalisation on the way out:
    - skills[].proficiency: map string levels to floats
    - experience_years: compute from career history if missing from profile
    - behavioural: map redrob_signals dict to BehaviouralSignals dataclass
    Never loads more than one record into memory at a time.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield _parse_candidate(raw)


def load_batch(data_path, batch_size=1000):
    """
    Yields lists of Candidate objects in batches of batch_size.
    Used for embedding generation which benefits from batching.
    """
    batch = []
    for cand in stream_candidates(data_path):
        batch.append(cand)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _parse_candidate(raw):
    profile = raw.get("profile", {})
    career_raw = raw.get("career_history", [])
    skills_raw = raw.get("skills", [])
    education_raw = raw.get("education", [])
    certs_raw = raw.get("certifications", [])
    languages_raw = raw.get("languages", [])
    signals_raw = raw.get("redrob_signals", {})

    # ── Career ──
    career = []
    for r in career_raw:
        role = CareerRole(
            title=r.get("title", ""),
            company=r.get("company", ""),
            duration_months=r.get("duration_months", 0) or 0,
            description=r.get("description", ""),
            raw=r,
        )
        career.append(role)

    # ── Skills ──
    skills = []
    for s in skills_raw:
        sk = Skill(
            name=s.get("name", ""),
            proficiency=PROFICIENCY_MAP.get(s.get("proficiency", ""), 0.0),
            duration_months=s.get("duration_months", 0) or 0,
            endorsements=s.get("endorsements", 0) or 0,
        )
        skills.append(sk)

    # ── Education (kept as dicts — not used in scoring per EDA Finding 6) ──
    education = []
    for e in education_raw:
        education.append({
            "institution": e.get("institution", ""),
            "degree": e.get("degree", ""),
            "field_of_study": e.get("field_of_study", ""),
            "start_year": e.get("start_year"),
            "end_year": e.get("end_year"),
            "grade": e.get("grade"),
            "tier": e.get("tier"),
        })

    # ── Certifications ──
    certifications = []
    for c in certs_raw:
        certifications.append({
            "name": c.get("name", ""),
            "issuer": c.get("issuer", ""),
            "year": c.get("year"),
        })

    # ── Languages ──
    languages = [l.get("language", "") for l in languages_raw]

    # ── Behavioural signals ──
    # Clean sentinel values: -1 means "no data", not a bad signal
    signals_clean = dict(signals_raw)
    for sentinel_key in ["github_activity_score", "offer_acceptance_rate"]:
        val = signals_clean.get(sentinel_key)
        if val is not None and (val == -1 or val == -1.0):
            signals_clean[sentinel_key] = None

    response_rate = signals_clean.get("recruiter_response_rate", 0.0) or 0.0
    last_active = signals_raw.get("last_active_date", "")
    is_active = _is_active_from_date(last_active)
    open_to_work = signals_clean.get("open_to_work_flag", False) or False
    notice_period = signals_clean.get("notice_period_days", 0) or 0
    interview_rate = signals_clean.get("interview_completion_rate", 0.0) or 0.0
    ghosting = signals_clean.get("applications_submitted_30d", 0) or 0

    behavioural = BehaviouralSignals(
        response_rate=float(response_rate),
        is_active=is_active,
        open_to_work=open_to_work,
        notice_period_days=int(notice_period),
        interview_completion_rate=float(interview_rate),
        ghosting_count=int(ghosting),
        raw=signals_clean,
    )

    # ── Candidate ──
    # Compute experience_years from career history if missing from profile
    exp = profile.get("years_of_experience")
    if exp is None:
        total_months = sum(r.duration_months for r in career)
        exp = total_months / 12.0

    candidate = Candidate(
        candidate_id=raw.get("candidate_id", ""),
        headline=profile.get("headline", ""),
        summary=profile.get("summary", ""),
        experience_years=float(exp) if exp else 0.0,
        location=profile.get("location", ""),
        career=career,
        skills=skills,
        education=education,
        certifications=certifications,
        languages=languages,
        behavioural=behavioural,
        raw=raw,
    )
    return candidate


def _is_active_from_date(last_active_str):
    """
    Active if last_active_date is within the recent window.
    EDA max last_active_date was 2026-05-27; threshold set to 90 days before.
    """
    if not last_active_str or len(last_active_str) < 10:
        return True
    try:
        year = int(last_active_str[:4])
        month = int(last_active_str[5:7])
        day = int(last_active_str[8:10])
        # Active if on or after 2026-02-26 (max - 90 days from EDA)
        if year > 2026:
            return True
        if year == 2026:
            if month > 2:
                return True
            if month == 2 and day >= 26:
                return True
        return False
    except (ValueError, TypeError, IndexError):
        return True
