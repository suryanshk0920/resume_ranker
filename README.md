# AI Candidate Ranking System

Ranks 100K candidates for a Senior AI Engineer role at a Series A startup. Built for the Redrob Hackathon.

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Place `candidates.jsonl` in `data/` (provided separately by the challenge).

## Run (single command)

```bash
python run.py
```

Produces `output/submission.csv` with 100 ranked candidates.

**No API key required.** The repo includes pre-computed LLM reasoning in `cache/reasoning.json`. The pipeline uses it automatically when no API key is detected.

### To regenerate reasoning with your own API key

Set one of these in `.env`:

| Provider | Key | Supported Models |
|----------|-----|-----------------|
| NVIDIA | `NVIDIA_API_KEY` | `minimaxai/minimax-m3`, `nvidia/llama-3.1-nemotron-70b-instruct` |
| Groq | `GROQ_API_KEY` | `llama-3.1-8b-instant`, `mixtral-8x7b-32768` |
| OpenRouter | `OPENROUTER_API_KEY` | `google/gemma-4-26b-a4b-it:free` |

Then run:
```bash
python run.py --fresh-reasoning
```

### Flags

| Flag | Description |
|------|-------------|
| `--fresh-reasoning` | Regenerate LLM reasoning (ignore cache) |
| `--skip-reasoning` | Skip Phase 4 entirely |
| `--force-recalibrate` | Rerun data scan (ignore cache) |
| `--dry-run` | Run all phases, don't write CSV |

### First-run note

The first run downloads `BAAI/bge-small-en-v1.5` (~90MB) and computes embeddings for top 2000 candidates (~83s). Subsequent runs load cached embeddings instantly (~0.4s). Embedding pre-computation and Phase 4 reasoning are offline steps; only Phases 1-3 (~12s) count toward the 5-minute ranking window.

## Architecture

Five-phase pipeline:

1. **Phase 0 — Calibration** (offline, cached): Scans candidate distributions, decomposes JD into skill clusters
2. **Phase 1 — Hard Filter** (online): Honeypot → behavioural gates → archetype scoring → experience band. 100K → ~2,888
3. **Phase 2 — Deep Scoring** (online): Pre-filters top 2000 by fast scores, computes 5 components (semantic, technical, founding fit, behavioural, career quality). → top 500
4. **Phase 3 — Precision Rerank** (online): Diversity penalty + founding fit tiebreaker. 500 → final 100
5. **Phase 4 — Reasoning** (offline, cached): LLM-generated recruiter-facing reasoning per candidate

## Scoring Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Semantic fit | 0.35 | Contrastive embedding (full JD text vs negative JD) |
| Technical depth | 0.32 | AI/ML skill relevance × proficiency × duration × endorsements |
| Founding team fit | 0.00* | Tiebreaker layer (score-band re-sort + floor swap) |
| Behavioural | 0.20 | Response rate, engagement, availability, reliability |
| Career quality | 0.13 | Product-company ratio, progression arc, tenure, experience band |

*\*Founding fit is excluded from weighted sum; applied as a tiebreaker in Phase 3.*

## Sandbox (Google Colab)

Run the full pipeline in your browser with zero setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suryanshk0920/resume_ranker/blob/main/sandbox/redrob_ranker.ipynb)

**Steps:**
1. Click the badge above
2. Click **Runtime → Run all**
3. Wait ~60s — the CSV downloads automatically

No Google Drive mount needed. No API key needed. Dataset downloads automatically.

## Requirements

- Python 3.10+
- 16 GB RAM
- CPU only (no GPU)
- Network only on first run (model download) and Phase 4 (optional)

## Project Structure

```
├── run.py                     # Single entry point
├── src/
│   ├── config.py              # All constants and weights
│   ├── models.py              # Dataclasses
│   ├── phase0_calibrate.py    # Data scan + JD decomposition
│   ├── phase1_filter.py       # Broad filter (4 gates)
│   ├── phase2_score.py        # Deep scoring (5 components)
│   ├── phase3_rerank.py       # Precision rerank + tiebreakers
│   ├── phase4_reason.py       # LLM reasoning generation
│   ├── scoring/
│   │   ├── semantic.py        # Contrastive embedding similarity
│   │   ├── technical.py       # Skill depth formula
│   │   ├── founding_fit.py    # Startup/scope/velocity
│   │   ├── behavioural.py     # 23-signal composite
│   │   └── career_quality.py  # Product vs services, progression
│   └── utils/
│       ├── loader.py          # JSONL streaming reader
│       ├── normaliser.py      # Score normalisation
│       └── honeypot.py        # Consistency checks
├── cache/                     # Pre-computed artifacts (reasoning.json included)
├── data/                      # candidates.jsonl goes here
├── output/                    # submission.csv lands here
├── requirements.txt
├── .env.example
└── submission_metadata.yaml
```
