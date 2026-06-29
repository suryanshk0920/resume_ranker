# AI Candidate Ranking System

<a href="https://colab.research.google.com/github/suryanshk0920/resume_ranker/blob/main/sandbox/redrob_ranker.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="250" style="max-width: 100%;">
</a>

Ranks 100K candidates for a Senior AI Engineer (Founding Team) role. Built for the Redrob Hackathon.

---

## For Judges — How to Run (3 steps)

**Step 1:** Click the blue "Open In Colab" button above ↗

**Step 2:** In Colab, click **Runtime → Run all**

**Step 3:** Wait ~60 seconds. The file `submission.csv` will download automatically.

That's it. No account needed. No setup needed. No API keys.

---

## Local Setup (alternative)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Place `candidates.jsonl` in `data/`.

```bash
python run.py
```

Produces `output/submission.csv` — 100 ranked candidates with recruiter-facing reasoning.

**No API key required.** Pre-computed LLM reasoning is included in the repo. The pipeline uses it automatically when no API key is detected.

### Regenerate reasoning (optional)

Set one of these in `.env` and run `python run.py --fresh-reasoning`:

| Provider | Key | Models |
|----------|-----|--------|
| NVIDIA | `NVIDIA_API_KEY` | `minimaxai/minimax-m3` |
| Groq | `GROQ_API_KEY` | `llama-3.1-8b-instant` |
| OpenRouter | `OPENROUTER_API_KEY` | `google/gemma-4-26b-a4b-it:free` |

### Flags

| Flag | What it does |
|------|-------------|
| `--fresh-reasoning` | Regenerate LLM reasoning (ignore cache) |
| `--skip-reasoning` | Skip Phase 4 entirely |
| `--force-recalibrate` | Rerun data scan (ignore cache) |
| `--dry-run` | Run all phases, don't write CSV |

### First-run note

First run downloads `BAAI/bge-small-en-v1.5` (~90MB) and computes embeddings (~83s). Subsequent runs skip this (cached). Only Phases 1-3 (~12s) count toward the 5-minute ranking window.

---

## Architecture

Five-phase pipeline:

1. **Phase 0 — Calibration** (offline, cached): Scans candidate distributions, decomposes JD into skill clusters
2. **Phase 1 — Hard Filter** (online): Honeypot → behavioural gates → archetype scoring → experience band. 100K → ~2,888
3. **Phase 2 — Deep Scoring** (online): Pre-filters top 2000 by fast scores, computes 5 components (semantic, technical, founding fit, behavioural, career quality). → top 500
4. **Phase 3 — Precision Rerank** (online): Diversity penalty + founding fit tiebreaker. 500 → final 100
5. **Phase 4 — Reasoning** (offline, cached): LLM-generated recruiter-facing reasoning per candidate

## Scoring Components

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Semantic fit | 0.35 | Per-role embedding similarity vs full JD text |
| Technical depth | 0.32 | AI/ML skill relevance × proficiency × duration × endorsements |
| Founding team fit | 0.00* | Tiebreaker (startup experience, scope breadth, career velocity) |
| Behavioural | 0.20 | Response rate, engagement, reliability |
| Career quality | 0.13 | Product vs services experience, promotion trajectory, tenure, experience band |

*\*Founding fit is applied as a Phase 3 tiebreaker, not a weighted score component.*

## Requirements

- Python 3.10+, 16 GB RAM, CPU only (no GPU)
- Network only on first run (model download) and Phase 4 reasoning (optional)

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
│   ├── scoring/               # Individual scoring modules
│   └── utils/                 # Loader, normaliser, honeypot
├── cache/                     # Pre-computed artifacts (all cached — zero compute)
├── sandbox/                   # Google Colab notebook
├── requirements.txt
└── submission_metadata.yaml
```
