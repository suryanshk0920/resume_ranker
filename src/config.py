import os
from dotenv import load_dotenv
load_dotenv()

# Paths
DATA_PATH = os.getenv("DATA_PATH", "data/candidates.jsonl")
CACHE_DIR = os.getenv("CACHE_DIR", "cache/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 256

# Phase 1 — hard filter thresholds
MIN_RESPONSE_RATE = 0.15
MIN_INTERVIEW_COMPLETION = 0.20
MAX_GHOSTING_COUNT = 3
MIN_EXPERIENCE_YEARS = 2.2
MAX_EXPERIENCE_YEARS = 20.0
MIN_ARCHETYPE_SCORE = 0.05
AI_TITLE_BASE_RATE = 0.009

# Phase 2 — scoring weights (must sum to 1.0)
# founding_fit removed from weighted sum — moved to Phase 3 as a tiebreaker layer
WEIGHTS = {
    "semantic":       0.35,
    "technical":      0.32,
    "founding_fit":   0.00,
    "behavioural":    0.20,
    "career_quality": 0.13,
}

# Phase 3 — founding_fit tiebreaker
FOUNDING_FIT_TIEBREAKER_BAND = 0.02
FOUNDING_FIT_SWAP_THRESHOLD = 0.03
FOUNDING_FIT_FLOOR = 0.40

# Experience band — scoring curve
EXP_OPTIMAL_YEARS = 5.1
EXP_PEAK_HALFWIDTH = 4.0

# Founding fit signals
STARTUP_EMPLOYEE_THRESHOLD = 50
MIN_SCOPE_BREADTH_FOR_BONUS = 3
FAST_VELOCITY_YEARS = 4.0

# Phase 3 — reranking
TOP_N_BROAD = 2000
TOP_N_PRECISION = 500
FINAL_TOP_N = 100
DIVERSITY_SAME_COMPANY_PENALTY = 0.05

# Phase 4 — reasoning (Groq API, with Gemini/OpenRouter as alternatives)
# Priority: Groq → Gemini → OpenRouter → template fallback
# Default model works with Groq free tier; override for other providers
REASONING_MODEL = os.getenv("REASONING_MODEL", "llama-3.1-8b-instant")
REASONING_MAX_TOKENS = 120   # per-candidate budget; multiplied by batch size for API call
REASONING_BATCH_SIZE = 10    # candidates per API call (fewer calls = faster)

# API keys (set at least one for Phase 4 reasoning)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Caches
JD_CACHE_PATH = "cache/jd_decomposed.json"
CALIBRATION_CACHE_PATH = "cache/calibration.json"
EMBEDDINGS_CACHE_PATH = "cache/candidate_embeddings.pkl"

# — Data-driven EDA findings (from analysis/findings.md) —

# SUSPICIOUS_SKILL_NAMES: skills where mean_endorsements > 200 in the full
# candidate pool. Across all 133 unique skills and ~960K skill entries, no
# skill exceeded a mean endorsement count of 7.6 in the top 50. The dataset
# has no inflated-endorsement skills — keyword stuffing manifests as skills
# with zero duration/endorsements, not inflated ones. List kept empty as a
# safety check; update if per-skill endorsement inflation is later detected.
SUSPICIOUS_SKILL_NAMES = []

# CORRELATED_SIGNAL_PAIRS: signal pairs with |Pearson r| > 0.6 in the
# behavioural signal analysis. All computed correlations were very weak:
#   recruiter_response_rate vs is_active:                r = 0.0324
#   recruiter_response_rate vs interview_completion_rate: r = 0.0360
#   notice_period_days vs open_to_work_flag:             r = -0.0014
# No pair exceeded the 0.6 threshold, meaning all five behavioural sub-
# components (response rate, activity, interview completion, notice period,
# open-to-work) capture independent signals. The behavioural score can
# safely include all of them without double-counting.
CORRELATED_SIGNAL_PAIRS = []

# HARD_EXCLUDE_TITLES: confirmed non-AI titles from EDA (Section 3).
# These 11 titles account for ~45,740 candidates (45.7% of pool).
# But — only hard-exclude if career_history contains zero technical roles.
# The JD explicitly warns about keyword-stuffer traps: a "Marketing Manager"
# with prior ML engineer roles is a fit. A "Marketing Manager" with only
# marketing roles is not.
# AI_ARCHETYPE_TITLES: titles that indicate direct AI/ML archetype experience.
# These get the highest archetype score (1.0 for current, 0.85 for past).
AI_ARCHETYPE_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "data scientist", "nlp engineer", "research scientist",
    "deep learning engineer", "llm engineer", "applied scientist",
    "ai research engineer", "computer vision engineer", "mlops engineer",
    "ai/ml engineer", "ml researcher", "ai researcher"
]

HARD_EXCLUDE_TITLES = [
    "hr manager", "accountant", "civil engineer", "mechanical engineer",
    "content writer", "graphic designer", "sales executive",
    "customer support", "operations manager", "marketing manager",
    "project manager",
]

# CONSULTING_FIRM_PATTERNS: companies where ALL roles = hard exclude
# EDA found 9,745 candidates (9.7%) whose every role is at one of these firms
# JD explicitly says: "People who have only worked at consulting firms...
# without prior product-company experience" are not a fit
CONSULTING_FIRM_PATTERNS = [
    "tcs", "tata consultancy", "infosys", "wipro", "cognizant",
    "accenture", "capgemini", "deloitte", "hcl", "tech mahindra",
    "mphasis", "hexaware", "mindtree", "l&t infotech", "ltimindtree",
    "persistent", "mastech", "niit",
]

# CALIBRATION_NOTES: EDA findings that shaped the ranking architecture.
# Every threshold in this file is traceable to one of these findings.
CALIBRATION_NOTES = (
    "1) Only 0.9% of candidates (856/100K) hold AI-relevant titles — the "
    "archetype filter must rely on career history semantics, not just "
    "current-title keyword matching, or the pool collapses to under 1K "
    "before scoring. 2) Behavioural signals are essentially uncorrelated "
    "(all r < 0.04) — response rate, activity, and interview completion "
    "each capture distinct dimensions of candidate reliability and can be "
    "used independently without double-counting risk. 3) 9.7% of candidates "
    "(9,745) have exclusively worked at IT services/consulting firms (TCS, "
    "Infosys, Wipro, etc.) — hard-exclude only if EVERY role is at one of "
    "these firms; one product-company role redeems. 4) Education fields are "
    "synthetic placeholders (Tier-3 Engineering College, Local Engineering "
    "College) with unrealistic 17.5% PhD rate — education removed from "
    "scoring as it adds noise, not signal. 5) Honeypot true count ~44, not "
    "~80 — only role-level inconsistencies (19 overlap + 25 overclaim), "
    "not skill-duration overlap (17,887 legitimate cases of continuous "
    "learning across jobs). 6) All behavioural signals independent — "
    "increased behavioural weight from 0.15 to 0.17 with no double-counting "
    "risk."
)
