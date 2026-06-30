# System Architecture

```mermaid
graph TB
    subgraph Input["Input"]
        A["100K Candidates<br/>(candidates.jsonl)"]
        B["Job Description<br/>(full JD text)"]
    end

    subgraph Phase0["Phase 0 — Calibration (offline, cached)"]
        C["Data Distribution Scan<br/>experience_years, endorsements,<br/>skill durations, title frequency"]
        D["JD Decomposition<br/>→ skill clusters<br/>→ seniority band<br/>→ founding signals"]
    end

    subgraph Phase1["Phase 1 — Broad Filter (online)"]
        E["Gate 1: Honeypot Detection<br/>chronological + skill sanity checks"]
        F["Gate 2: Consulting-only Exclusion"]
        G["Gate 3: Archetype Scoring<br/>4-tier AI/ML title matching"]
        H["Gate 4: Experience Band<br/>2.2 – 20.0 years"]
    end

    subgraph Phase2["Phase 2 — Deep Scoring (online)"]
        I["4 Fast Scores<br/>technical + founding_fit +<br/>behavioural + career_quality"]
        J["Pre-select Top 2000"]
        K["Contrastive Semantic Embedding<br/>BAAI/bge-small-en-v1.5<br/>positive JD − 0.4 × negative JD"]
        L["Weighted Sum<br/>0.35×sem + 0.32×tech +<br/>0.20×behav + 0.13×career"]
    end

    subgraph Phase3["Phase 3 — Precision Rerank (online)"]
        M["Founding Fit Tiebreaker<br/>score-band re-sort ±0.02"]
        N["Diversity Check<br/>max 2/company in top 10"]
        O["Founding Fit Floor<br/>swap top-10 < 0.40"]
    end

    subgraph Phase4["Phase 4 — Reasoning (offline, cached)"]
        P["LLM (NVIDIA/Groq/OpenRouter)<br/>or Cached reasoning.json"]
    end

    subgraph Output["Output"]
        Q["submission.csv<br/>100 ranked candidates<br/>candidate_id, rank, score, reasoning"]
    end

    A --> C
    B --> D
    C --> E
    D --> E
    E --> F --> G --> H
    H --> I --> J --> K --> L
    L --> M --> N --> O
    O --> P --> Q
```

---

## Quick Reference — Per-Slide Breakdown

| Slide | Content |
|---|---|
| **Title** | Resume Ranker — Candidate Ranking for Founding AI Role |
| **Problem** | Rank 100K candidates with CPU-only, 5-min budget |
| **Architecture** | Paste the rendered Mermaid diagram |
| **Key Heuristics** | MIN_RESPONSE_RATE=0.15, MIN_ARCHETYPE_SCORE=0.05, DIVERSITY_PENALTY=0.05, TIEBREAKER_BAND=0.02 |
| **Weighted Score** | semantic=0.35, technical=0.32, behavioural=0.20, career_quality=0.13 |
| **Honeypot Detection** | 929 planted, 0 in top-100 |
| **Results** | 100 candidates ranked, fully cached, runs in ~30s |

---

## Rendering for Slides

1. Open https://mermaid.live
2. Paste the diagram (content between ` ```mermaid ` and ` ``` `)
3. Export as PNG — paste into your slide deck
