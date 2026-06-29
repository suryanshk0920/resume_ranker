# AI Candidate Ranking System — Complete Build Specification
> For agentic IDE execution. Follow every section in order. Do not skip, reorder, or modify any step. Every decision here is deliberate.

---

## 0. Meta-rules for the IDE

- Read this entire document before writing a single line of code.
- Every file listed in the File Tree must be created exactly as specified.
- Every function signature listed must be implemented exactly as specified.
- No external API calls during ranking runtime (Phase 1–3). The Anthropic API is called only in Phase 0 (JD decomposition, offline) and Phase 4 (reasoning generation, offline). Both produce cached JSON files before ranking begins.
- No GPU assumed. All compute must run on CPU only.
- Total runtime budget for the ranking pipeline (Phases 1–3): under 5 minutes on a machine with 16 GB RAM.
- Python version: 3.10+
- All dependencies must be installable via `pip` and be free/open-source.
- Do not install PyTorch with CUDA. Use the CPU-only build explicitly.

---

## 1. Repository Structure

Create this exact directory and file tree. Do not add extra files.

```
candidate-ranker/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   └── .gitkeep                      # candidates.jsonl goes here at runtime
│
├── cache/
│   └── .gitkeep                      # all generated cache files land here
│
├── output/
│   └── .gitkeep                      # submission.csv lands here
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # all constants and weight definitions
│   ├── models.py                     # dataclasses for Candidate, Score, Result
│   │
│   ├── phase0_calibrate.py           # data distribution scan + JD decomposition
│   ├── phase1_filter.py              # honeypot + behavioural gate + archetype filter
│   ├── phase2_score.py               # 5-component deep scoring
│   ├── phase3_rerank.py              # NDCG-aware rerank + diversity check
│   ├── phase4_reason.py              # batch LLM reasoning generation
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── semantic.py               # per-role embedding + cosine similarity
│   │   ├── technical.py              # skill depth formula
│   │   ├── founding_fit.py           # startup proxy + scope breadth + velocity
│   │   ├── behavioural.py            # RedRob 23-signal composite
│   │   └── career_quality.py         # product-co vs services, progression arc
│   │
│   └── utils/
│       ├── __init__.py
│       ├── loader.py                 # streaming JSONL reader
│       ├── normaliser.py             # score normalisation using Phase 0 bounds
│       └── honeypot.py              # all consistency checks
│
└── run.py                            # single entry point — runs all phases in order
```

---

## 2. Tech Stack

Every library listed here must be in `requirements.txt`. No others are needed.

```
# requirements.txt — exact contents

# Core data
jsonlines==4.0.0
numpy==1.26.4
pandas==2.2.2

# Embeddings (CPU-only sentence transformer)
sentence-transformers==3.0.1
torch==2.3.1+cpu  # CPU build — important
torchvision==0.18.1+cpu

# Similarity and scoring
scikit-learn==1.5.0

# LLM API (Phase 0 and Phase 4 only — offline preprocessing)
anthropic==0.28.0

# Environment variable management
python-dotenv==1.0.1

# Progress bars
tqdm==4.66.4
```

Install command to include in README:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Embedding model:** `BAAI/bge-small-en-v1.5`
- Why: 33M parameters, fits in 512 MB RAM, runs on CPU in ~2ms per sentence, outperforms `all-MiniLM-L6-v2` on retrieval benchmarks.
- Downloaded automatically by `sentence-transformers` on first run, then cached locally in `~/.cache/huggingface/`.
- Do not use any other embedding model.

---

## 3. Environment Configuration

### `.env.example`
```
ANTHROPIC_API_KEY=your_key_here
DATA_PATH=data/candidates.jsonl
CACHE_DIR=cache/
OUTPUT_DIR=output/
```

### `.gitignore`
```
.env
cache/*.json
cache/*.pkl
output/*.csv
data/*.jsonl
__pycache__/
*.pyc
.DS_Store
```

---

## 4. Data Models (`src/models.py`)

Define these dataclasses exactly. Every downstream module imports from here.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CareerRole:
    title: str
    company: str
    duration_months: int
    description: str
    is_startup: bool = False          # set by founding_fit.py
    scope_breadth: int = 0            # count of distinct skill domains used

@dataclass
class Skill:
    name: str
    proficiency: float                # normalised 0.0-1.0
    duration_months: int
    endorsements: int

@dataclass
class BehaviouralSignals:
    response_rate: float
    is_active: bool
    open_to_work: bool
    notice_period_days: int
    interview_completion_rate: float
    ghosting_count: int
    # raw dict of all 23 signals also stored
    raw: dict = field(default_factory=dict)

@dataclass
class Candidate:
    candidate_id: str
    headline: str
    summary: str
    experience_years: float
    location: str
    career: list[CareerRole]
    skills: list[Skill]
    education: list[dict]
    certifications: list[dict]
    languages: list[str]
    behavioural: BehaviouralSignals
    raw: dict = field(default_factory=dict)   # original JSON preserved

@dataclass
class ComponentScores:
    semantic: float = 0.0
    technical: float = 0.0
    founding_fit: float = 0.0
    behavioural: float = 0.0
    career_quality: float = 0.0

@dataclass
class CandidateResult:
    candidate_id: str
    rank: int
    score: float
    component_scores: ComponentScores
    reasoning: str = ""
    honeypot_flag: bool = False
    gate_fail_reason: str = ""        # non-empty means filtered out
```

---

## 5. Config (`src/config.py`)

All tunable constants live here. IDE must not hardcode any of these values inline in other files.

```python
import os
from dotenv import load_dotenv
load_dotenv()

# Paths
DATA_PATH = os.getenv("DATA_PATH", "data/candidates.jsonl")
CACHE_DIR = os.getenv("CACHE_DIR", "cache/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 256            # number of texts per encoding batch

# Phase 1 — hard filter thresholds
MIN_RESPONSE_RATE = 0.15
MIN_INTERVIEW_COMPLETION = 0.20
MAX_GHOSTING_COUNT = 3
MIN_EXPERIENCE_YEARS = 2.0
MAX_EXPERIENCE_YEARS = 20.0
MIN_ARCHETYPE_SCORE = 0.10           # candidates below this are excluded

# Phase 2 — scoring weights (must sum to 1.0)
WEIGHTS = {
    "semantic":       0.30,
    "technical":      0.25,
    "founding_fit":   0.20,
    "behavioural":    0.15,
    "career_quality": 0.10,
}

# Experience band — scoring curve
EXP_OPTIMAL_YEARS = 7.0
EXP_PEAK_HALFWIDTH = 4.0             # years either side of optimal for full score

# Founding fit signals
STARTUP_EMPLOYEE_THRESHOLD = 50      # company size to count as "startup"
MIN_SCOPE_BREADTH_FOR_BONUS = 3      # distinct skill domains per role
FAST_VELOCITY_YEARS = 4.0            # years-to-senior threshold for velocity bonus

# Phase 3 — reranking
TOP_N_BROAD = 2000                   # Phase 1 output target
TOP_N_PRECISION = 500                # Phase 2 output target
FINAL_TOP_N = 100                    # final submission count
DIVERSITY_SAME_COMPANY_PENALTY = 0.05  # per duplicate company in top 10

# Phase 4 — reasoning
REASONING_MODEL = "claude-sonnet-4-6"
REASONING_MAX_TOKENS = 150
REASONING_BATCH_SIZE = 10            # candidates per API call batch

# JD decomposition
JD_MODEL = "claude-sonnet-4-6"
JD_CACHE_PATH = "cache/jd_decomposed.json"
CALIBRATION_CACHE_PATH = "cache/calibration.json"
EMBEDDINGS_CACHE_PATH = "cache/candidate_embeddings.pkl"
```

---

## 6. Phase 0 — Calibration (`src/phase0_calibrate.py`)

**Purpose:** Run once before ranking. Produces two cache files that all later phases depend on.

### 6.1 Data distribution scan

Function signature:
```python
def run_calibration(data_path: str, cache_path: str) -> dict:
    """
    Stream candidates.jsonl once. Compute and cache:
    - experience_years: p10, p25, p50, p75, p90
    - endorsement counts: p50, p90 per skill
    - skill duration months: p50, p90
    - title_frequency: dict of title → count (top 200)
    - behavioural signal ranges: min, max, mean for all 23 signals
    Returns calibration dict. Writes to cache_path as JSON.
    """
```

Implementation notes:
- Use `jsonlines` to stream — do not load all 100k records into memory at once.
- Collect values in running arrays using `reservoir_sampling(k=10000)` for memory safety on large fields.
- Write output as JSON to `CALIBRATION_CACHE_PATH`.
- If cache file already exists and `--force` flag not passed, load and return cached version. Do not recompute.

### 6.2 JD Decomposition

The job description text is hardcoded as a constant in this file (paste the full JD text as a Python string). Do not read it from disk.

Function signature:
```python
def decompose_jd(jd_text: str, cache_path: str) -> dict:
    """
    Call Anthropic API once with the JD text.
    Returns structured dict. Writes to cache_path.
    If cache exists, return cached version without API call.
    """
```

Prompt to send (use this exactly):
```
You are parsing a job description for a candidate ranking system.

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

Return only valid JSON. No explanation. No markdown fences.
```

Parse the response as JSON. If parsing fails, retry once. If it fails again, raise `ValueError` with the raw response.

---

## 7. Utilities

### 7.1 `src/utils/loader.py`

```python
def stream_candidates(data_path: str):
    """
    Generator. Yields Candidate dataclass instances one at a time.
    Reads candidates.jsonl using jsonlines library.
    Performs field normalisation on the way out:
    - skills[].proficiency: map string levels to floats
      {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    - experience_years: compute from career history if missing from profile
    - behavioural: map redrob_signals dict to BehaviouralSignals dataclass
    Never loads more than one record into memory at a time.
    """

def load_batch(data_path: str, batch_size: int = 1000):
    """
    Yields lists of Candidate objects in batches of batch_size.
    Used for embedding generation which benefits from batching.
    """
```

### 7.2 `src/utils/honeypot.py`

```python
def check_chronological_consistency(candidate: Candidate) -> tuple[bool, str]:
    """
    Returns (is_clean, reason).
    Checks:
    1. sum(role.duration_months for role in career) <= total_career_span_months + 12
       (allow 12-month overlap for job transitions)
    2. No skill with duration_months > total_career_span_months
    3. If any role has Senior/Lead/Principal in title, check that
       (graduation_year + 3) <= role_start_year (minimum 3 years post-grad)
    4. No role start year before candidate's plausible entry (graduation - 1 year)
    Returns (False, reason_string) if any check fails.
    Returns (True, "") if all pass.
    """

def check_skill_duration_sanity(candidate: Candidate) -> tuple[bool, str]:
    """
    Returns (is_clean, reason).
    For each skill: skill.duration_months must be <= candidate total career months.
    Any skill with duration_months == 0 AND endorsements == 0 gets flagged as
    a potential keyword-stuffer (does not disqualify, but reduces skill score to 0).
    Returns (False, reason) if any skill duration exceeds career length.
    """

def compute_honeypot_score(candidate: Candidate) -> float:
    """
    Returns a float 0.0 (clean) to 1.0 (almost certainly honeypot).
    Aggregates soft signals:
    - Skills with 0 duration + 0 endorsements: +0.1 per skill, max 0.5
    - Profile summary that mentions more than 15 distinct AI buzzwords 
      with no corresponding career history: +0.3
    - Experience years implied by career > stated profile experience by > 3 years: +0.2
    Does not exclude — just informs the gate in phase1_filter.py.
    """
```

### 7.3 `src/utils/normaliser.py`

```python
def normalise_score(value: float, p10: float, p90: float) -> float:
    """
    Min-max normalise using p10 as floor and p90 as ceiling.
    Clamp output to [0.0, 1.0].
    Formula: (value - p10) / (p90 - p10), clamped.
    """

def experience_band_score(years: float, optimal: float, halfwidth: float) -> float:
    """
    Gaussian-shaped scoring curve.
    Returns 1.0 at optimal, decays symmetrically by halfwidth.
    Formula: exp(-0.5 * ((years - optimal) / halfwidth) ** 2)
    Clamp output to [0.0, 1.0].
    """
```

---

## 8. Phase 1 — Broad Filter (`src/phase1_filter.py`)

**Input:** stream from `loader.py` — all 100k candidates  
**Output:** list of `Candidate` objects, target ~2,000, hard max 5,000  
**Runtime budget:** 60 seconds

```python
def run_phase1(data_path: str, calibration: dict) -> list[Candidate]:
    """
    Streams all candidates. Applies four sequential gates.
    A candidate is excluded if any gate returns False.
    Returns list of passing candidates.
    Prints: total read, honeypot excluded, gate1 excluded, gate2 excluded,
            gate3 excluded, gate4 excluded, passing count.
    """
```

### Gate 1 — Honeypot elimination (hard exclude)

Call `check_chronological_consistency()` and `check_skill_duration_sanity()`.  
If either returns `(False, reason)`: exclude candidate. Log to `cache/excluded_honeypot.jsonl`.

Also call `compute_honeypot_score()`. If score >= 0.6: exclude.

### Gate 2 — Behavioural hard gate (hard exclude)

Exclude if ALL of these are simultaneously true (AND, not OR):
```python
candidate.behavioural.response_rate < MIN_RESPONSE_RATE
and not candidate.behavioural.is_active
```

Also exclude if:
```python
candidate.behavioural.interview_completion_rate < MIN_INTERVIEW_COMPLETION
and candidate.behavioural.ghosting_count > MAX_GHOSTING_COUNT
```

### Gate 3 — Role archetype filter

```python
def archetype_score(candidate: Candidate) -> float:
    """
    Score 0.0 to 1.0 based on title pattern matching.

    AI/ML archetype titles (score 1.0 each, take max across career):
    ["machine learning", "ml engineer", "ai engineer", "data scientist",
     "nlp engineer", "research scientist", "deep learning", "computer vision",
     "llm", "applied scientist", "mlops", "ai researcher", "data engineer",
     "software engineer" with AI/ML keywords in description]

    Disqualifying titles (score 0.0, hard exclude if current role):
    ["marketing manager", "sales", "accountant", "civil engineer",
     "hr manager", "content writer", "graphic designer", "business analyst"
     (without technical qualifications)]

    Weight current role 2x, previous roles 1x each.
    Normalise to 0.0-1.0.
    """
```

Exclude if `archetype_score(candidate) < MIN_ARCHETYPE_SCORE`.

### Gate 4 — Experience band hard floor/ceiling

Exclude if `candidate.experience_years < MIN_EXPERIENCE_YEARS`.  
Exclude if `candidate.experience_years > MAX_EXPERIENCE_YEARS`.  
(Soft penalties for sub-optimal but within-range values applied in Phase 2.)

---

## 9. Phase 2 — Deep Scoring (`src/phase2_score.py`)

**Input:** ~2,000 candidates from Phase 1  
**Output:** list of `CandidateResult`, top 500 by final score  
**Runtime budget:** 90 seconds

```python
def run_phase2(
    candidates: list[Candidate],
    jd_decomposed: dict,
    calibration: dict
) -> list[CandidateResult]:
    """
    Computes all 5 component scores for each candidate.
    Combines with WEIGHTS from config.
    Returns sorted list, top TOP_N_PRECISION results.
    """
```

### 9.1 Semantic Score (`src/scoring/semantic.py`)

```python
def build_jd_embedding(jd_decomposed: dict) -> np.ndarray:
    """
    Construct JD embedding text from decomposed fields:
    - required_skill_clusters joined as a sentence
    - founding_team_signals joined
    - nice_to_have_signals joined
    Encode with SentenceTransformer(EMBEDDING_MODEL).
    Cache result to EMBEDDINGS_CACHE_PATH + '_jd.pkl'.
    Return embedding vector.
    """

def build_candidate_embeddings(
    candidates: list[Candidate]
) -> dict[str, np.ndarray]:
    """
    For each candidate, create per-role embeddings (not one profile blob).
    Text per role: f"{role.title} at {role.company}: {role.description}"
    Encode all role texts in one batched call for efficiency.
    Aggregate per candidate using recency weighting:
      weights = [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05]
      (index 0 = most recent role)
    weighted_avg = sum(weight_i * embedding_i) / sum(weights used)
    Also add the candidate headline embedding at weight 0.5.
    Cache full results dict to EMBEDDINGS_CACHE_PATH.
    Return dict: candidate_id -> aggregated embedding vector.
    """

def compute_semantic_scores(
    candidates: list[Candidate],
    jd_embedding: np.ndarray,
    candidate_embeddings: dict[str, np.ndarray]
) -> dict[str, float]:
    """
    Cosine similarity between each candidate embedding and JD embedding.
    Returns dict: candidate_id -> score (0.0-1.0).
    """
```

### 9.2 Technical Depth Score (`src/scoring/technical.py`)

```python
# AI/ML skill keywords — match against skill names (case-insensitive substring)
AI_SKILL_KEYWORDS = [
    "python", "pytorch", "tensorflow", "transformers", "huggingface",
    "llm", "nlp", "embeddings", "vector", "rag", "retrieval", "ranking",
    "recommendation", "deep learning", "neural", "bert", "gpt", "fine-tuning",
    "mlops", "kubernetes", "docker", "fastapi", "sql", "spark", "distributed",
    "reinforcement learning", "computer vision", "ocr", "speech", "multimodal",
    "langchain", "pinecone", "faiss", "weaviate", "elasticsearch"
]

def skill_relevance_weight(skill_name: str) -> float:
    """
    Returns 1.0 if skill_name matches any AI_SKILL_KEYWORDS.
    Returns 0.3 for general engineering skills (git, linux, agile, etc.)
    Returns 0.1 for unrelated skills.
    Use case-insensitive substring matching.
    """

def compute_technical_score(candidate: Candidate, calibration: dict) -> float:
    """
    For each skill:
      1. relevance = skill_relevance_weight(skill.name)
      2. prof_weight = skill.proficiency  # already normalised 0.0-1.0
      3. dur_ceil = min(skill.duration_months / 36.0, 1.0)  # 3 yrs = full credit
      4. end_weight = log(skill.endorsements + 1) / log(calibration['endorsement_p90'] + 1)
         clamped to [0.0, 1.0]
      5. skill_score = relevance * min(prof_weight, dur_ceil) * (0.6 + 0.4 * end_weight)
         (endorsements add up to 40% bonus, not a gate)

    Zero-duration + zero-endorsement skills: skill_score = 0.0 always.

    candidate_technical_score = mean(top_10_skill_scores by individual value)
    Normalise using calibration bounds.
    Return float 0.0-1.0.
    """
```

### 9.3 Founding Team Fit (`src/scoring/founding_fit.py`)

```python
# Known large-company indicators (if company matches, it is NOT a startup)
LARGE_COMPANY_PATTERNS = [
    "google", "microsoft", "amazon", "meta", "apple", "netflix", "uber",
    "airbnb", "salesforce", "oracle", "ibm", "accenture", "deloitte",
    "infosys", "wipro", "tcs", "cognizant", "capgemini"
]

def is_startup_company(company_name: str) -> bool:
    """
    Returns True if company does NOT match any LARGE_COMPANY_PATTERNS.
    Case-insensitive matching.
    Not perfect — intentionally uses absence-of-known-large-companies as proxy.
    """

def compute_scope_breadth(role: CareerRole, all_skills: list[Skill]) -> int:
    """
    Count distinct high-level skill domains present in role.description
    and role title. Domains:
    ["ml/ai", "backend/infra", "data engineering", "research/publications",
     "leadership/management", "product/strategy", "frontend", "devops/cloud"]
    Return count of domains matched (0-8).
    """

def compute_seniority_velocity(candidate: Candidate) -> float:
    """
    Find year of first Senior/Lead/Principal/Staff/Director title in career.
    velocity_years = year_of_first_senior - estimated_graduation_year
    If no senior title found: return 0.5 (neutral).
    Score: if velocity_years <= FAST_VELOCITY_YEARS: 1.0
           elif velocity_years <= FAST_VELOCITY_YEARS * 2: 0.6
           else: 0.3
    """

def compute_founding_fit_score(candidate: Candidate) -> float:
    """
    Combines three sub-scores:
    
    1. startup_score (weight 0.4):
       Count roles at is_startup companies / total roles.
       Bonus +0.2 if any role at company with <20 employees (proxy: very small name).

    2. scope_score (weight 0.35):
       Mean scope_breadth across all roles, normalised to 0.0-1.0 (max breadth = 8).
       Bonus +0.15 if any single role has breadth >= MIN_SCOPE_BREADTH_FOR_BONUS.

    3. velocity_score (weight 0.25):
       From compute_seniority_velocity().

    founding_fit_score = 0.4*startup_score + 0.35*scope_score + 0.25*velocity_score
    Clamp to [0.0, 1.0].
    Return float.
    """
```

### 9.4 Behavioural Reliability (`src/scoring/behavioural.py`)

```python
def compute_behavioural_score(candidate: Candidate, calibration: dict) -> float:
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
```

### 9.5 Career Quality (`src/scoring/career_quality.py`)

```python
def compute_career_quality_score(candidate: Candidate) -> float:
    """
    Three sub-scores:

    1. product_company_score (weight 0.45):
       Ratio of roles at product companies vs. services/consulting companies.
       Product companies: tech companies building their own product.
       Services companies: "consulting", "services", "solutions", "outsourcing" in name.
       Score = product_roles / total_roles.

    2. progression_score (weight 0.35):
       Detect upward title progression across career.
       Score 1.0 if clear seniority increase across at least 3 roles.
       Score 0.6 if lateral movement only.
       Score 0.2 if apparent downward movement.
       Detect by checking for Junior -> Mid -> Senior -> Lead -> Principal pattern.

    3. tenure_score (weight 0.20):
       Mean role duration across all roles.
       Optimal: 18-36 months per role.
       < 6 months average: 0.2 (job hopper signal)
       6-12 months: 0.5
       12-36 months: 1.0
       > 48 months: 0.7 (too slow-moving for founding team)

    career_quality_score = 0.45*product + 0.35*progression + 0.20*tenure
    Clamp to [0.0, 1.0].
    """
```

### 9.6 Score Combination

```python
def combine_scores(component_scores: ComponentScores) -> float:
    """
    Weighted sum using WEIGHTS from config.py.
    final = (
        WEIGHTS['semantic']       * component_scores.semantic +
        WEIGHTS['technical']      * component_scores.technical +
        WEIGHTS['founding_fit']   * component_scores.founding_fit +
        WEIGHTS['behavioural']    * component_scores.behavioural +
        WEIGHTS['career_quality'] * component_scores.career_quality
    )
    Clamp to [0.0, 1.0].
    Return float.
    """
```

---

## 10. Phase 3 — Precision Rerank (`src/phase3_rerank.py`)

**Input:** top 500 `CandidateResult` objects from Phase 2  
**Output:** final top 100 `CandidateResult` objects with ranks assigned  
**Runtime budget:** 60 seconds

```python
def run_phase3(results: list[CandidateResult]) -> list[CandidateResult]:
    """
    1. Sort by score descending.
    2. Apply diversity penalty to top 10:
       For each pair in top 10 with same company: apply DIVERSITY_SAME_COMPANY_PENALTY
       to the lower-ranked one. Re-sort top 10 only after penalties.
    3. Assign final ranks 1-100.
    4. Validate: assert no candidate appears twice.
    5. Validate: assert no honeypot-flagged candidate in top 100.
    6. Return final list of 100 CandidateResult, rank field populated.
    """

def apply_diversity_check(top10: list[CandidateResult]) -> list[CandidateResult]:
    """
    Within top 10, detect company clusters.
    If >2 candidates from same company, penalise the 3rd+ by DIVERSITY_SAME_COMPANY_PENALTY.
    Also check education: if >3 from same university, apply same penalty.
    Re-sort and return adjusted top 10.
    """
```

---

## 11. Phase 4 — Reasoning Generation (`src/phase4_reason.py`)

**Input:** final 100 `CandidateResult` objects, JD decomposed dict  
**Output:** same list with `reasoning` field populated  
**Runtime budget:** 60 seconds (runs offline, not in the 5-min ranking window)

```python
def generate_reasoning_batch(
    results: list[CandidateResult],
    candidates_map: dict[str, Candidate],
    jd_decomposed: dict
) -> list[CandidateResult]:
    """
    For each candidate in results:
    1. Build a compact context string from the candidate's data:
       - Current title and company
       - Most recent 2 roles (title, company, key skills used)
       - Top 3 AI-relevant skills with proficiency
       - experience_years
       - Their top 2 scoring component names and values
    2. Call Anthropic API with the prompt below.
    3. Populate result.reasoning with the response.
    Process in batches of REASONING_BATCH_SIZE.
    Handle rate limit errors with exponential backoff (2s, 4s, 8s, max 3 retries).
    """

REASONING_PROMPT = """
You are writing shortlist notes for a recruiter reviewing candidates for a Senior AI Engineer role at a Series A startup (founding team hire).

JD requires: {required_skill_clusters}
Founding team signals needed: {founding_team_signals}

Candidate data:
{candidate_context}

Their top scoring factors: {top_score_factors}

Write exactly 2 sentences explaining why this candidate fits (or the strongest elements of their fit). 
Rules:
- Reference a specific role or project from their history, not generic attributes.
- Mention at least one founding-team-relevant signal (startup experience, scope breadth, fast growth, shipped something from scratch).
- Do not use the word "passionate". Do not use phrases like "strong background" or "proven track record".
- Write in third person. No bullet points. No markdown.
- Maximum 60 words total.
"""
```

---

## 12. Entry Point (`run.py`)

```python
#!/usr/bin/env python3
"""
Single entry point. Run: python run.py
Flags:
  --force-recalibrate   : ignore cached calibration, rerun Phase 0 data scan
  --force-jd            : ignore cached JD decomposition, call API again
  --skip-reasoning      : skip Phase 4 (submit without LLM reasoning)
  --dry-run             : run all phases but do not write output CSV
"""

import argparse
import time
from src.config import *
from src.phase0_calibrate import run_calibration, decompose_jd
from src.phase1_filter import run_phase1
from src.phase2_score import run_phase2
from src.phase3_rerank import run_phase3
from src.phase4_reason import generate_reasoning_batch
from src.utils.loader import stream_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recalibrate", action="store_true")
    parser.add_argument("--force-jd", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    print("[Phase 0] Calibration...")
    calibration = run_calibration(DATA_PATH, CALIBRATION_CACHE_PATH, force=args.force_recalibrate)
    jd = decompose_jd(JD_TEXT, JD_CACHE_PATH, force=args.force_jd)
    print(f"  Done in {time.time()-t0:.1f}s")

    t1 = time.time()
    print("[Phase 1] Broad filter...")
    filtered = run_phase1(DATA_PATH, calibration)
    print(f"  {len(filtered)} candidates passing. Done in {time.time()-t1:.1f}s")

    t2 = time.time()
    print("[Phase 2] Deep scoring...")
    scored = run_phase2(filtered, jd, calibration)
    print(f"  Top {len(scored)} scored. Done in {time.time()-t2:.1f}s")

    t3 = time.time()
    print("[Phase 3] Precision rerank...")
    final = run_phase3(scored)
    print(f"  Final {len(final)} ranked. Done in {time.time()-t3:.1f}s")

    if not args.skip_reasoning:
        t4 = time.time()
        print("[Phase 4] Generating reasoning...")
        candidates_map = {c.candidate_id: c for c in filtered}
        final = generate_reasoning_batch(final, candidates_map, jd)
        print(f"  Done in {time.time()-t4:.1f}s")

    total = time.time() - t0
    print(f"\nTotal pipeline time: {total:.1f}s")

    if not args.dry_run:
        write_submission(final)
        print(f"Submission written to {OUTPUT_DIR}submission.csv")

def write_submission(results):
    import csv, os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "submission.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "candidate_id": r.candidate_id,
                "rank": r.rank,
                "score": round(r.score, 4),
                "reasoning": r.reasoning
            })

if __name__ == "__main__":
    main()
```

---

## 13. Output Format

The file `output/submission.csv` must have exactly these columns, exactly this order:

```
candidate_id,rank,score,reasoning
cand_00001,1,0.9241,"Led the retrieval ranking system at a 15-person AI startup that served 2M daily queries, directly matching the LLM-powered search infrastructure this role needs to build from scratch. Joined as employee #4 and grew the ML team to 8 engineers, demonstrating the founding-team operational range the JD explicitly calls out."
cand_00042,2,0.9187,"..."
...
```

Rules:
- Exactly 100 rows (not counting header).
- `rank` is an integer 1-100, no duplicates, sequential.
- `score` is a float rounded to 4 decimal places.
- `reasoning` is a quoted string, max 60 words, no internal double quotes (escape if present).
- File encoding: UTF-8.

---

## 14. README.md

Write a README with these sections exactly:

```markdown
# AI Candidate Ranking System

## Setup
1. Clone the repo
2. Install dependencies:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
3. Copy .env.example to .env and add your ANTHROPIC_API_KEY
4. Place candidates.jsonl in data/

## Running
python run.py

## Flags
--force-recalibrate   Rerun data distribution scan (ignore cache)
--force-jd            Redecompose JD via API (ignore cache)  
--skip-reasoning      Skip LLM reasoning generation
--dry-run             Run pipeline but do not write CSV

## Architecture
Five-phase pipeline:
- Phase 0: Calibration + JD decomposition (offline, cached)
- Phase 1: Hard filter — 100k → ~2,000 candidates
- Phase 2: Deep scoring — 2,000 → top 500
- Phase 3: Precision rerank — 500 → top 100
- Phase 4: LLM reasoning generation (offline, cached)

## Output
output/submission.csv — 100 ranked candidates with reasoning
```

---

## 15. Implementation Order for the IDE

Execute in this exact sequence. Do not proceed to the next step until the current step passes its verification check.

| Step | File | Verification |
|------|------|-------------|
| 1 | Create directory structure + `.gitkeep` files | `ls` confirms all dirs exist |
| 2 | `requirements.txt` + install | `pip install` exits 0 |
| 3 | `.env.example`, `.gitignore` | Files present |
| 4 | `src/models.py` | `python -c "from src.models import Candidate"` exits 0 |
| 5 | `src/config.py` | `python -c "from src.config import WEIGHTS; assert sum(WEIGHTS.values()) == 1.0"` |
| 6 | `src/utils/loader.py` | Unit test: stream 10 records, confirm Candidate objects returned |
| 7 | `src/utils/honeypot.py` | Unit test: feed a candidate with skill duration > career length, confirm False returned |
| 8 | `src/utils/normaliser.py` | Unit test: normalise_score(50, 10, 90) == 0.5 |
| 9 | `src/phase0_calibrate.py` | Run on 1000 records, confirm JSON cache written |
| 10 | `src/scoring/semantic.py` | Confirm JD embedding is shape (384,) for bge-small |
| 11 | `src/scoring/technical.py` | Unit test: zero-duration zero-endorsement skill scores 0.0 |
| 12 | `src/scoring/founding_fit.py` | Unit test: large company returns is_startup=False |
| 13 | `src/scoring/behavioural.py` | Unit test: response_rate=0, ghosting=5 gives low score |
| 14 | `src/scoring/career_quality.py` | Unit test: all consulting roles gives low product_company_score |
| 15 | `src/phase1_filter.py` | Run on 1000 records, confirm exclusion counts printed |
| 16 | `src/phase2_score.py` | Run on 100 candidates, confirm sorted results returned |
| 17 | `src/phase3_rerank.py` | Confirm 100 results returned, ranks 1-100, no duplicates |
| 18 | `src/phase4_reason.py` | Run on 5 candidates, confirm reasoning populated, max 60 words |
| 19 | `run.py` | Full pipeline run with `--dry-run`, confirm no errors, time < 300s |
| 20 | Full run with real data | `output/submission.csv` exists, 100 rows, valid format |

---

## 16. Critical Rules — Do Not Violate

1. **Never embed the full candidate profile as one string.** Always embed per-role separately and aggregate.
2. **Never use skill score > 0 when both duration_months == 0 and endorsements == 0.** These are keyword stuffers.
3. **Never use behavioural signals as a multiplicative modifier at the end.** They are a hard gate in Phase 1 and a soft score component in Phase 2.
4. **Never call the Anthropic API inside the ranking loop.** Only in Phase 0 and Phase 4, both offline.
5. **Never load all 100k candidates into memory at once.** Stream with `jsonlines` in Phase 1.
6. **Never hardcode weights inline.** All weights come from `config.py`.
7. **Never output more or fewer than 100 rows in submission.csv.**
8. **Never assign the same rank to two candidates.**
9. **Embedding cache must be written after Phase 2 and read on subsequent runs.** Do not recompute embeddings if cache exists and `--force` not passed.
10. **The scoring metric is NDCG@10 (50%) + NDCG@50 (30%) + MAP (15%) + P@10 (5%).** This means positions 1-10 matter more than 11-100. Ensure Phase 3 applies maximum precision to the top 10 slots.

---

## 17. Known Edge Cases — Handle These

| Edge case | Handling |
|-----------|----------|
| Candidate has zero career roles | Set experience_years = 0, all career-based scores = 0. Keep if skills qualify. |
| Candidate summary/description is empty string | Skip empty fields in embedding construction. Do not embed empty strings. |
| `duration_months` field is null/missing on a skill | Treat as 0 |
| `endorsements` field is null/missing | Treat as 0 |
| Phase 1 passes fewer than 500 candidates | Lower `MIN_ARCHETYPE_SCORE` threshold by 0.05 and retry. Log the adjustment. |
| Phase 1 passes more than 5000 candidates | Tighten archetype score threshold by 0.05 and retry. Log. |
| Anthropic API call fails in Phase 4 | Fill reasoning with empty string "". Do not crash the pipeline. |
| Candidate has roles with future dates | Flag as potential honeypot. Exclude if honeypot_score >= 0.6. |
| Duplicate candidate_id in input | Keep first occurrence. Log duplicate count. |

---

## 18. Files the IDE Must NOT Create

- No Jupyter notebooks
- No Flask/FastAPI server
- No Docker files
- No CI/CD configuration
- No test framework files (pytest etc.) — unit tests in this spec are manual inline checks only
- No frontend/UI code
- No database files

This is a command-line data pipeline. Nothing else.