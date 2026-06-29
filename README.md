# AI Candidate Ranking System

Ranks 100K candidates for a Senior AI Engineer role at a Series A startup using career history semantics. Built for the Redrob Hackathon 2025.

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your OpenRouter API key (optional — Phase 4 uses template fallback if unset):
   ```bash
   cp .env.example .env
   ```
4. Place `candidates.jsonl` in `data/`

## Running

Single command to produce `output/submission.csv`:

```bash
python run.py
```

### Flags

| Flag | Description |
|------|-------------|
| `--force-recalibrate` | Rerun data distribution scan (ignore cache) |
| `--force-jd` | Redecompose JD via API (ignore cache) |
| `--skip-reasoning` | Skip LLM reasoning generation |
| `--dry-run` | Run pipeline but do not write CSV |

### Pre-computation note

Phase 2 embeds candidates using `BAAI/bge-small-en-v1.5`. The first run downloads the model (~90MB) and computes embeddings for top 2000 candidates (~83s). Subsequent runs load cached embeddings instantly (~0.4s). The 5-minute ranking window covers only Phases 1-3; embedding pre-computation and Phase 4 reasoning are offline steps.

## Output

`output/submission.csv` — 100 ranked candidates with columns:

```
candidate_id,rank,score,reasoning
```

- Exactly 100 rows, ranks 1-100 (no gaps, no duplicates)
- Score non-increasing with rank (ties: break by candidate_id ascending)
- Reasoning: 2-sentence recruiter-facing note, max 60 words

## Architecture

Five-phase pipeline:

1. **Phase 0 — Calibration** (offline, cached): Scans candidate data distributions and decomposes JD into skill clusters, seniority band, founding team signals
2. **Phase 1 — Hard Filter** (online): Applies honeypot elimination, behavioural gates, archetype scoring (4 tiers), and experience band filter. 100K → ~2,888
3. **Phase 2 — Deep Scoring** (online): Pre-filters top 2000 by fast non-semantic scores, then computes 5 component scores (semantic, technical, founding fit, behavioural, career quality). ~2,888 → top 500
4. **Phase 3 — Precision Rerank** (online): Applies diversity penalty for same-company clusters in top 10, assigns final ranks 1-100. 500 → top 100
5. **Phase 4 — Reasoning Generation** (offline, cached): Calls OpenRouter API (free model) to generate recruiter-facing reasoning. Falls back to template when API unavailable.

## Scoring Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Semantic fit | 0.30 | Per-role cosine similarity with JD embedding |
| Technical depth | 0.25 | AI/ML skill relevance × proficiency × duration × endorsements |
| Founding team fit | 0.20 | Startup experience, scope breadth, career velocity |
| Behavioural | 0.17 | Response rate, engagement, availability, reliability |
| Career quality | 0.08 | Product-company ratio, progression arc, tenure pattern |

## Requirements

- Python 3.10+
- 16 GB RAM
- CPU only (no GPU)
- Network only during Phase 4 (optional — template fallback works offline)

## Project Structure

```
├── run.py                     # Single entry point
├── src/
│   ├── config.py              # All constants and weights
│   ├── models.py              # Dataclasses
│   ├── phase0_calibrate.py    # Data scan + JD decomposition
│   ├── phase1_filter.py       # Broad filter (4 gates)
│   ├── phase2_score.py        # Deep scoring (5 components)
│   ├── phase3_rerank.py       # Precision rerank + diversity
│   ├── phase4_reason.py       # LLM reasoning generation
│   ├── scoring/
│   │   ├── semantic.py        # Per-role embedding similarity
│   │   ├── technical.py       # Skill depth formula
│   │   ├── founding_fit.py    # Startup/scope/velocity
│   │   ├── behavioural.py     # 23-signal composite
│   │   └── career_quality.py  # Product vs services, progression
│   └── utils/
│       ├── loader.py          # JSONL streaming reader
│       ├── normaliser.py      # Score normalisation
│       └── honeypot.py        # Consistency checks
├── cache/                     # Pre-computed artifacts
├── data/                      # candidates.jsonl goes here
├── output/                    # submission.csv lands here
├── requirements.txt
├── .env.example
└── submission_metadata.yaml
```
