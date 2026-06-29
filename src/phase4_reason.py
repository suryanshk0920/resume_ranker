"""
Phase 4 — Reasoning Generation.
Generates recruiter-facing reasoning for each candidate using OpenRouter API
(compatible with free models). Falls back to template-based reasoning when
API key is not set or API calls fail. Results cached to cache/reasoning.json.
Runs offline — not part of the 5-minute ranking window.
"""

import json
import os
import time

from src.config import (
    REASONING_MODEL, REASONING_MAX_TOKENS,
    REASONING_BATCH_SIZE, CACHE_DIR,
)

REASONING_CACHE_PATH = os.path.join(CACHE_DIR, "reasoning.json")

BATCH_REASONING_PROMPT = """You are writing shortlist notes for a recruiter reviewing candidates for a Senior AI Engineer role at a Series A startup (founding team hire).

JD requires: {required_skill_clusters}
Founding team signals needed: {founding_team_signals}

Below are {count} candidates. For EACH candidate, write exactly 2 sentences explaining why they fit (or the strongest elements of their fit).

Rules:
- Sentence 1: Lead with a specific project, system, or achievement they shipped — not their title or company.
- Sentence 2: Explain what sets this candidate apart from other AI engineers. This must be unique per candidate — do NOT reuse the same reason across different candidates.
- Mention at least one founding-team-relevant signal (startup experience, scope breadth, fast career growth, shipped from scratch).
- Reference their component scores if relevant (e.g. "semantic fit=0.92", "career quality=0.84").
- Do not use the word "passionate". Do not use "strong background" or "proven track record".
- Write in third person. No bullet points. No markdown.
- Maximum 100 words per candidate.
- Never repeat the same closing phrase across different candidates.

Output each candidate on its own line using the format:
CAND_ID|reasoning text

{candidates_section}"""


def _load_cache():
    if os.path.exists(REASONING_CACHE_PATH):
        with open(REASONING_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(REASONING_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _build_short_context(cand, result):
    """Compact candidate summary for batch prompt."""
    top_roles = cand.career[:2]
    ai_skills = sorted(
        [s for s in cand.skills if s.proficiency >= 0.5],
        key=lambda s: -(s.proficiency * s.duration_months)
    )[:3]
    parts = [
        f"Location: {cand.location}",
        f"Current: {cand.headline}",
        f"Experience: {cand.experience_years:.1f} years, {len(cand.skills)} skills",
    ]
    for i, role in enumerate(top_roles):
        label = "Most recent" if i == 0 else "Previous"
        parts.append(f"{label}: {role.title} at {role.company} ({role.duration_months}mo)")
    if ai_skills:
        skill_str = "; ".join(f"{s.name} ({s.proficiency:.0%}, {s.duration_months}mo)" for s in ai_skills)
        parts.append(f"Top skills: {skill_str}")
    # Component scores
    cs = result.component_scores
    parts.append(f"Scores: sem={cs.semantic:.2f} tech={cs.technical:.2f} founding={cs.founding_fit:.2f} behav={cs.behavioural:.2f} career={cs.career_quality:.2f}")
    if cand.behavioural.notice_period_days <= 30:
        parts.append(f"Notice period: {cand.behavioural.notice_period_days} days")
    return "\n".join(parts)


def _build_score_factors(result):
    scores = {
        "semantic fit":      result.component_scores.semantic,
        "technical depth":   result.component_scores.technical,
        "founding team fit": result.component_scores.founding_fit,
        "behavioural":       result.component_scores.behavioural,
        "career quality":    result.component_scores.career_quality,
    }
    top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
    return "; ".join(f"{k}={v:.2f}" for k, v in top2)


def _template_reasoning(cand, score_factors):
    """Template fallback when API is unavailable."""
    top_role = cand.career[0] if cand.career else None
    ai_skills = [s.name for s in cand.skills if s.proficiency >= 0.5][:2]
    parts = [cand.headline.replace("|", "-").strip()]
    if top_role:
        parts.append(f"{top_role.duration_months}m at {top_role.company}")
    if ai_skills:
        parts.append("skilled in " + ", ".join(ai_skills))
    if cand.behavioural.response_rate >= 0.5:
        parts.append("responsive to recruiters")
    if cand.behavioural.is_active:
        parts.append("recently active")
    if score_factors:
        parts.append(f"top factors: {score_factors}")
    return "; ".join(parts)


def _call_openrouter(prompt, client, model):
    """Make API calls with retry on rate limits. Sleeps for Groq's suggested delay."""
    import re, time
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=REASONING_MAX_TOKENS * REASONING_BATCH_SIZE,
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("API returned empty response (content is None)")
            return content.strip()
        except Exception as e:
            is_429 = (getattr(e, 'status_code', None) == 429 or
                      getattr(e, 'code', None) == 429 or
                      "429" in str(e) or
                      "rate_limit" in str(e).lower())
            if not is_429:
                raise
            match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(e))
            wait = float(match.group(1)) + 1 if match else 3.0
            print(f"    rate limited, retrying in {wait:.0f}s...")
            time.sleep(wait)
    raise RuntimeError("API call failed after 5 retries due to rate limits")


def _parse_batch_response(response_text, batch_ids):
    """
    Parse API response into {cand_id: reasoning} dict.
    Expected format: each line is CAND_ID|reasoning text
    """
    result = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "|" not in line:
            continue
        cand_id, reasoning = line.split("|", 1)
        cand_id = cand_id.strip()
        reasoning = reasoning.strip()
        if cand_id not in batch_ids:
            continue
        words = reasoning.split()
        if len(words) > 60:
            reasoning = " ".join(words[:60])
        result[cand_id] = reasoning
    return result


def generate_reasoning_batch(results, candidates_map, jd_decomposed, use_cache=False):
    """
    For each of the top 100 results, call OpenRouter API to generate
    recruiter-facing reasoning. Processes candidates in batches of
    REASONING_BATCH_SIZE to reduce API calls.
    Falls back to template per-batch when API calls fail.
    When use_cache=True, loads previous results from cache/reasoning.json
    and skips API for cached candidates.
    """
    cache = {}
    if use_cache:
        cache = _load_cache()

    # Fill cached results if use_cache is enabled
    uncached = []
    for result in results:
        if use_cache and result.candidate_id in cache:
            result.reasoning = cache[result.candidate_id]
        else:
            uncached.append(result)

    if not uncached:
        return results

    required = ", ".join(jd_decomposed.get("required_skill_clusters", []))
    founding = ", ".join(jd_decomposed.get("founding_team_signals", []))

    # Try to initialise API client — priority: Groq → Gemini → OpenRouter
    client = None
    active_model = REASONING_MODEL
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                timeout=30.0,
                max_retries=0,
            )
            print(f"  Groq client ready, will try model: {active_model}")
        except Exception as e:
            print(f"  Failed to create Groq client: {type(e).__name__}")

    if client is None and gemini_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
                timeout=30.0,
                max_retries=0,
            )
            print(f"  Gemini client ready, will try model: {active_model}")
        except Exception as e:
            print(f"  Failed to create Gemini client: {type(e).__name__}")

    if client is None and openrouter_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                timeout=30.0,
                max_retries=0,
            )
            print(f"  OpenRouter client ready, will try model: {active_model}")
        except Exception as e:
            print(f"  Failed to create OpenRouter client: {type(e).__name__}")

    updated = False
    num_batches = (len(uncached) + REASONING_BATCH_SIZE - 1) // REASONING_BATCH_SIZE

    for batch_idx in range(num_batches):
        start = batch_idx * REASONING_BATCH_SIZE
        batch = uncached[start:start + REASONING_BATCH_SIZE]
        batch_ids = {r.candidate_id for r in batch}

        # Build batch prompt section
        lines = []
        for result in batch:
            cand = candidates_map.get(result.candidate_id)
            if not cand:
                continue
            context = _build_short_context(cand, result)
            score_factors = _build_score_factors(result)
            lines.append(f"{result.candidate_id}\n{context}\nTop factors: {score_factors}")

        candidates_section = "\n\n".join(
            f"{j+1}. {line}" for j, line in enumerate(lines)
        )

        prompt = BATCH_REASONING_PROMPT.format(
            required_skill_clusters=required,
            founding_team_signals=founding,
            count=len(batch),
            candidates_section=candidates_section,
        )

        # Try API call
        api_ok = False
        if client is not None and active_model is not None:
            try:
                response_text = _call_openrouter(prompt, client, active_model)
                parsed = _parse_batch_response(response_text, batch_ids)
                if parsed:
                    for cand_id, reasoning in parsed.items():
                        cache[cand_id] = reasoning
                        updated = True
                    api_ok = True
            except Exception as e:
                print(f"  API error for batch {batch_idx+1}/{num_batches}: {e}")

        if not api_ok:
            print(f"  Batch {batch_idx+1}/{num_batches}: using template fallback")
            for result in batch:
                cand = candidates_map.get(result.candidate_id)
                if cand:
                    cache[result.candidate_id] = _template_reasoning(cand, _build_score_factors(result))
                    updated = True

        print(f"  Reasoned {min(start + REASONING_BATCH_SIZE, len(uncached))}/{len(uncached)} candidates...")

    # Fill all results from cache
    for result in results:
        result.reasoning = cache.get(result.candidate_id, "")

    if use_cache and updated:
        _save_cache(cache)
        print(f"  Reasoning cache updated: {REASONING_CACHE_PATH}")

    return results
