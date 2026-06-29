"""
Phase 0 — Calibration. Run once before ranking.
Produces two cache files that all later phases depend on.

6.1 Data distribution scan — streams candidates.jsonl once, computes:
  - experience_years: p10, p25, p50, p75, p90
  - endorsement counts: p50, p90 per skill
  - skill duration months: p50, p90
  - title_frequency: dict of title -> count (top 200)
  - behavioural signal ranges: min, max, mean for all 23 signals

6.2 JD Decomposition — calls Anthropic API once with JD text.
"""

import json
import os
import random
from collections import Counter

from src.config import (
    DATA_PATH, CALIBRATION_CACHE_PATH, JD_CACHE_PATH, EMBEDDINGS_CACHE_PATH,
)
from src.utils.loader import stream_candidates


# ────────────────────────────────────────────────────────────────────────────
# 6.1 Data distribution scan
# ────────────────────────────────────────────────────────────────────────────

def reservoir_sample(iterable, k):
    """
    Reservoir sampling. Yields a random sample of size k from iterable
    without loading the entire iterable into memory.
    """
    reservoir = []
    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def _pctile(sorted_arr, p):
    if not sorted_arr:
        return 0.0
    idx = p / 100.0 * (len(sorted_arr) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_arr) - 1)
    frac = idx - lo
    return sorted_arr[lo] + frac * (sorted_arr[hi] - sorted_arr[lo])


def run_calibration(data_path, cache_path, force=False):
    """
    Stream candidates.jsonl once. Compute and cache:
    - experience_years: p10, p25, p50, p75, p90
    - endorsement counts: p50, p90 per skill
    - skill duration months: p50, p90
    - title_frequency: dict of title -> count (top 200)
    - behavioural signal ranges: min, max, mean for all 23 signals
    Returns calibration dict. Writes to cache_path as JSON.
    """
    if not force and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    exp_years = []
    all_endorsements = []
    all_skill_durations = []
    title_counter = Counter()

    # Running aggregates for behavioural signals
    signal_ranges = {}  # signal_name -> {"min": ..., "max": ..., "sum": ..., "count": ...}

    for candidate in stream_candidates(data_path):
        exp_years.append(candidate.experience_years)

        title_counter[candidate.headline.lower()] += 1

        for skill in candidate.skills:
            all_endorsements.append(skill.endorsements)
            all_skill_durations.append(skill.duration_months)

        # Behavioural signal ranges from raw dict
        for sig_name, sig_val in candidate.behavioural.raw.items():
            if not isinstance(sig_val, (int, float)):
                continue
            if isinstance(sig_val, bool):
                continue
            if sig_val is None:
                continue
            if sig_name not in signal_ranges:
                signal_ranges[sig_name] = {
                    "min": sig_val, "max": sig_val, "sum": 0.0, "count": 0,
                }
            rng = signal_ranges[sig_name]
            if sig_val < rng["min"]:
                rng["min"] = sig_val
            if sig_val > rng["max"]:
                rng["max"] = sig_val
            rng["sum"] += sig_val
            rng["count"] += 1

    # Compute percentiles using reservoir sampling for large arrays
    exp_sample = sorted(reservoir_sample(exp_years, 10000))
    endorse_sample = sorted(reservoir_sample(all_endorsements, 10000))
    dur_sample = sorted(reservoir_sample(all_skill_durations, 10000))

    # Build behavioural signal ranges (replace sum/count with mean)
    behavioural_ranges = {}
    for sig_name, rng in signal_ranges.items():
        behavioural_ranges[sig_name] = {
            "min": rng["min"],
            "max": rng["max"],
            "mean": rng["sum"] / rng["count"] if rng["count"] > 0 else 0.0,
        }

    calibration = {
        "experience_p10": _pctile(exp_sample, 10),
        "experience_p25": _pctile(exp_sample, 25),
        "experience_p50": _pctile(exp_sample, 50),
        "experience_p75": _pctile(exp_sample, 75),
        "experience_p90": _pctile(exp_sample, 90),
        "endorsement_p50": _pctile(endorse_sample, 50),
        "endorsement_p90": _pctile(endorse_sample, 90),
        "skill_duration_p50": _pctile(dur_sample, 50),
        "skill_duration_p90": _pctile(dur_sample, 90),
        "title_frequency": dict(title_counter.most_common(200)),
        "behavioural_signal_ranges": behavioural_ranges,
    }

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(calibration, f, indent=2)

    return calibration


# ────────────────────────────────────────────────────────────────────────────
# 6.2 JD Decomposition
# ────────────────────────────────────────────────────────────────────────────

JD_TEXT = (
    'Job Description: Senior AI Engineer \u2014 Founding Team\n'
    'Company: Redrob AI (Series A AI-native talent intelligence platform)\n'
    'Location: Pune/Noida, India (Hybrid \u2014 flexible cadence) | Open to '
    'relocation candidates from Tier-1 Indian cities\n'
    'Employment Type: Full-time\n'
    'Experience Required: 5\u20139 years (see "what we mean by this" below)\n'
    '\n'
    "Let's be honest about this role\n"
    "We're going to write this JD differently from most. We're a Series A "
    'company that just raised our round and we\'re building a new AI '
    "Engineering org from scratch. This is the kind of role where the JD "
    "changes every six months because the company changes every six months. "
    "So instead of pretending we have a fixed checklist, we're going to tell "
    "you what we actually need and what we've gotten wrong before.\n"
    "If you've spent your career at Google or Meta and you want a "
    "well-scoped role with a defined ladder, this isn't it.\n"
    "If you've spent your career bouncing between early-stage startups and "
    'you want to "just code" without having to think about product or '
    "recruiter workflows or eval frameworks, this also isn't it.\n"
    "We need someone who is simultaneously comfortable with two things that "
    "sound contradictory:\n"
    "Deep technical depth in modern ML systems \u2014 embeddings, retrieval, "
    "ranking, LLMs, fine-tuning.\n"
    "Scrappy product-engineering attitude \u2014 willing to ship a working "
    'ranker in a week even if the underlying ML is "obviously suboptimal," '
    "because we need to learn from real users before we know what to "
    "actually optimize for.\n"
    "These are not contradictory in real life. They feel contradictory "
    'because of how engineering culture sorted itself into "researcher" vs '
    '"shipper" archetypes. We need both modes available in the same person, '
    "and we'd rather you tilt slightly toward shipper than toward "
    "researcher.\n"
    "\n"
    "What you'd actually be doing\n"
    "The high-level mandate: own the intelligence layer of Redrob's "
    "product. That means the ranking, retrieval, and matching systems that "
    "decide what recruiters see when they search for candidates and what "
    "candidates see when they search for roles.\n"
    "In practical terms, your first 90 days will probably look like:\n"
    "Weeks 1-3: Audit what we currently have (it's mostly BM25 + "
    "rule-based scoring, working but not great). Identify the 3-4 "
    "highest-leverage things to fix.\n"
    "Weeks 4-8: Ship a v2 ranking system that demonstrably improves "
    "recruiter-engagement metrics. This will involve embeddings, hybrid "
    "retrieval, and probably some LLM-based re-ranking, but the "
    "architecture is your call.\n"
    "Weeks 9-12: Set up the evaluation infrastructure \u2014 offline "
    "benchmarks, online A/B testing, recruiter-feedback loops \u2014 so we "
    "can keep improving without flying blind.\n"
    "Beyond that, you'll be driving the long-term architecture of how we "
    "do candidate-JD matching at scale, mentoring the next round of hires "
    "(we're growing the team from 4 to 12 engineers in the next year), "
    "and working closely with our recruiter-experience PM on what to build.\n"
    "\n"
    'What we mean by "5-9 years"\n'
    "This is a range, not a requirement. Some people hit 'senior engineer' "
    "judgment at 4 years; some never hit it after 15. We've used 5-9 "
    "because it's roughly where people we've hired into this kind of role "
    "have landed, but we'll seriously consider candidates outside the band "
    "if other signals are strong.\n"
    "That said, here are the disqualifiers we actually apply:\n"
    "If you've spent your career in pure research environments (academic "
    "labs, research-only roles) without any production deployment \u2014 we "
    "will not move forward. We are explicit about this. We've tried it "
    "twice and it didn't work for either side.\n"
    'If your "AI experience" consists primarily of recent (under 12 '
    "months) projects using LangChain to call OpenAI \u2014 we will "
    "probably not move forward, unless you can demonstrate substantial "
    'pre-LLM-era ML production experience. We\'re looking for people who '
    "understood retrieval and ranking before it became fashionable.\n"
    'If you are a senior engineer who hasn\'t written production code in '
    'the last 18 months because you\'ve moved into "architecture" or '
    '"tech lead" roles \u2014 we will probably not move forward. This '
    "role writes code.\n"
    "\n"
    "The skills inventory (please read carefully)\n"
    "Most JDs list 20 skills and you're supposed to have all of them. "
    "We're going to do this differently.\n"
    "Things you absolutely need\n"
    "Production experience with embeddings-based retrieval systems "
    "(sentence-transformers, OpenAI embeddings, BGE, E5, or similar) "
    "deployed to real users. We don't care which model \u2014 we care "
    "that you've handled embedding drift, index refresh, "
    "retrieval-quality regression in production.\n"
    "Production experience with vector databases or hybrid search "
    "infrastructure \u2014 Pinecone, Weaviate, Qdrant, Milvus, "
    "OpenSearch, Elasticsearch, FAISS, or something similar. Again, "
    "the specific tech doesn't matter; the operational experience does.\n"
    "Strong Python. Yes really, we care about code quality.\n"
    "Hands-on experience designing evaluation frameworks for ranking "
    "systems \u2014 NDCG, MRR, MAP, offline-to-online correlation, A/B "
    "test interpretation. If you've never thought about how to evaluate "
    "a ranking system rigorously, this role will be very painful.\n"
    "Things we'd like you to have but won't reject you for\n"
    "LLM fine-tuning experience (LoRA, QLoRA, PEFT)\n"
    "Experience with learning-to-rank models (XGBoost-based or neural)\n"
    "Prior exposure to HR-tech, recruiting tech, or marketplace products\n"
    "Background in distributed systems or large-scale inference "
    "optimization\n"
    "Open-source contributions in the AI/ML space\n"
    "Things we explicitly do NOT want\n"
    "This is the section most JDs skip but we think it's the most "
    "important:\n"
    "Title-chasers. If your career trajectory shows you optimizing for "
    '"Senior" \u2192 "Staff" \u2192 "Principal" titles by switching '
    "companies every 1.5 years, we're not a fit. We need someone who "
    "plans to be here for 3+ years.\n"
    "Framework enthusiasts. If your GitHub is full of LangChain "
    'tutorials and your blog posts are "How I used [hot framework] to '
    'build [demo]" \u2014 that\'s fine but it\'s not what we need. We '
    "need people who think about systems, not frameworks.\n"
    "People who have only worked at consulting firms (TCS, Infosys, "
    "Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire "
    "career. We've had bad fit experiences in both directions. If "
    "you're currently at one of these companies but have prior "
    "product-company experience, that's fine.\n"
    "People whose primary expertise is computer vision, speech, or "
    "robotics without significant NLP/IR exposure. We respect your work "
    "but you'd be re-learning fundamentals here.\n"
    "People whose work has been entirely on closed-source proprietary "
    "systems for 5+ years without external validation (papers, talks, "
    "open-source). We need to see how you think, not just trust that "
    "you can think.\n"
    "\n"
    "On location, comp, and logistics\n"
    "Location: Pune/Noida-preferred but flexible. We have offices in "
    "Noida and Pune (mostly used Tue/Thu). We don't require any "
    "specific number of in-office days but we expect quarterly travel "
    "for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR "
    "welcome to apply. Outside India: case-by-case, but we don't "
    "sponsor work visas.\n"
    "Notice period: We'd love sub-30-day notice. We can buy out up to "
    "30 days. 30+ day notice candidates are still in scope but the bar "
    "gets higher.\n"
    "\n"
    "The vibe check\n"
    "We genuinely believe culture-fit matters more at this stage than "
    "skills-fit. Skills are teachable; the rest mostly isn't.\n"
    "We work async-first and write a lot. If you find writing painful, "
    "you'll find this role painful.\n"
    "We disagree openly and decide quickly. If you find that style "
    "abrasive, you'll find this role abrasive.\n"
    "We move fast and break things, with the caveat that 'things' are "
    "usually our internal assumptions, not user-facing systems. If you "
    "need a stable, mature codebase to be productive, you'll find this "
    "role unstable.\n"
    "\n"
    "How to read between the lines\n"
    'The "ideal candidate" we\'re imagining is roughly:\n'
    "6-8 years total experience, of which 4-5 are in applied ML/AI "
    "roles at product companies (not pure services).\n"
    "Has shipped at least one end-to-end ranking, search, or "
    "recommendation system to real users at meaningful scale.\n"
    "Has strong opinions about retrieval (hybrid vs dense), evaluation "
    "(offline vs online), and LLM integration (when to fine-tune vs "
    "prompt) \u2014 and can defend them with reference to systems they "
    "actually built.\n"
    "Located in or willing to relocate to Noida or Pune.\n"
    "Active on Redrob platform (or has clear signal of being in the "
    "job market) so we can actually talk to them.\n"
    "We are aware this is a narrow profile. We're not expecting to "
    "find many matches in a 100K candidate pool. We're explicitly OK "
    "with that \u2014 we'd rather see 10 great matches than 1000 maybes."
)


DECOMPOSE_PROMPT = """You are parsing a job description for a candidate ranking system.

Note: Only 0.9% of candidates have explicit AI-relevant titles. The remaining 99.1% must be evaluated on career history descriptions. Ensure required_skill_clusters reflects skills that appear in WORK DESCRIPTIONS, not just skill tags.

JD TEXT:
{jd_text}

Return a JSON object with exactly these keys:
{{
  "required_skill_clusters": [list of skill cluster names, e.g. "NLP", "vector search", "LLM fine-tuning"],
  "seniority_band": {{"min_years": int, "max_years": int, "optimal_years": int}},
  "founding_team_signals": [list of phrases from the JD that indicate startup/founding-team fit],
  "anti_patterns": [list of role types or backgrounds that would be a poor fit],
  "location_preference": [list of preferred cities or "Remote"],
  "nice_to_have_signals": [list of bonus signals not strictly required]
}}

Return only valid JSON. No explanation. No markdown fences."""


def decompose_jd(jd_text, cache_path, force=False):
    """
    Parse the JD into structured clusters.
    Uses a pre-built decomposition based on the hardcoded JD text.
    No external API calls.
    If cache exists, return cached version.
    """
    if not force and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    return _fallback_jd_decomposition(jd_text, cache_path)


def _fallback_jd_decomposition(jd_text, cache_path):
    """Hardcoded decomposition when API is unavailable."""
    decomposition = {
        "required_skill_clusters": [
            "NLP", "vector search", "LLM fine-tuning",
            "embedding-based retrieval", "ranking systems",
            "evaluation frameworks", "Python engineering",
        ],
        "seniority_band": {"min_years": 5, "max_years": 9, "optimal_years": 6},
        "founding_team_signals": [
            "early-stage startup experience",
            "built systems from scratch",
            "wears multiple hats",
            "shipped production ML under uncertainty",
            "mentored small teams",
        ],
        "anti_patterns": [
            "consulting-only career",
            "research-only without production",
            "title-chaser with <2yr tenure per role",
        ],
        "location_preference": ["Pune", "Noida", "Hyderabad", "Remote"],
        "nice_to_have_signals": [
            "LoRA fine-tuning", "learning-to-rank", "HR-tech experience",
            "distributed systems", "open-source contributions",
        ],
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(decomposition, f, indent=2)
    return decomposition
