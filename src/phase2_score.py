"""
Phase 2 — Deep Scoring.
Computes all 5 component scores for each candidate from Phase 1 output.
"""

import os
import pickle

from src.config import WEIGHTS, TOP_N_BROAD, TOP_N_PRECISION, EMBEDDINGS_CACHE_PATH
from src.models import ComponentScores, CandidateResult
from src.scoring.technical import compute_technical_score
from src.scoring.founding_fit import compute_founding_fit_score
from src.scoring.behavioural import compute_behavioural_score
from src.scoring.career_quality import compute_career_quality_score


def combine_scores(component_scores):
    """Weighted sum using WEIGHTS from config.py. founding_fit removed — it's a Phase 3 tiebreaker."""
    score = (
        WEIGHTS["semantic"] * component_scores.semantic
        + WEIGHTS["technical"] * component_scores.technical
        + WEIGHTS["behavioural"] * component_scores.behavioural
        + WEIGHTS["career_quality"] * component_scores.career_quality
    )
    return max(0.0, min(1.0, score))


def run_phase2(candidates, jd_decomposed, calibration):
    """Score all candidates, return top TOP_N_PRECISION."""
    if not candidates:
        return []

    # Phase 1: compute 4 fast scores for ALL candidates (no embeddings yet)
    fast_scores = []
    for cand in candidates:
        tech = compute_technical_score(cand, calibration)
        founding = compute_founding_fit_score(cand)
        beh = compute_behavioural_score(cand, calibration)
        career = compute_career_quality_score(cand)
        fast_scores.append((cand, tech, founding, beh, career))

    # Pre-score without semantic to pick top candidates for embedding (founding_fit excluded — it's a tiebreaker)
    pre_weights = {k: WEIGHTS[k] for k in WEIGHTS if k not in ("semantic", "founding_fit")}
    pws = sum(pre_weights.values())
    for i, (cand, tech, founding, beh, career) in enumerate(fast_scores):
        pre = (pre_weights["technical"] * tech
               + pre_weights["behavioural"] * beh + pre_weights["career_quality"] * career)
        fast_scores[i] = (cand, tech, founding, beh, career, pre / pws if pws > 0 else 0.0)

    # Take top TOP_N_BROAD for embedding
    fast_scores.sort(key=lambda x: -x[5])
    top_for_embed = [x[0] for x in fast_scores[:TOP_N_BROAD]]

    # Embed only top candidates — check cache first to avoid loading SentenceTransformer
    jd_cache = EMBEDDINGS_CACHE_PATH + "_jd.pkl"
    if os.path.exists(jd_cache) and os.path.exists(EMBEDDINGS_CACHE_PATH):
        with open(jd_cache, "rb") as f:
            jd_emb = pickle.load(f)
        with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
            cand_embs = pickle.load(f)
        from src.scoring.semantic import compute_semantic_scores
    else:
        from src.scoring.semantic import build_jd_embedding, build_candidate_embeddings, compute_semantic_scores
        jd_emb = build_jd_embedding(jd_decomposed)
        cand_embs = build_candidate_embeddings(top_for_embed)
    semantic_map = compute_semantic_scores(top_for_embed, jd_emb, cand_embs)

    # Full scoring for all candidates (0.0 semantic for non-embedded)
    results = []
    for cand, tech, founding, beh, career, _ in fast_scores:
        sem = semantic_map.get(cand.candidate_id, 0.0)
        comp = ComponentScores(
            semantic=sem, technical=tech, founding_fit=founding,
            behavioural=beh, career_quality=career,
        )
        final_score = round(combine_scores(comp), 4)
        top_role = cand.career[0] if cand.career else None
        results.append(CandidateResult(
            candidate_id=cand.candidate_id,
            rank=0,
            score=final_score,
            component_scores=comp,
            company=(top_role.company or "") if top_role else "",
        ))

    results.sort(key=lambda r: (-r.score, r.candidate_id))
    for i, r in enumerate(results):
        r.rank = i + 1

    top = results[:TOP_N_PRECISION]

    # Write audit CSV
    import csv, os
    from src.config import OUTPUT_DIR
    audit_path = os.path.join(OUTPUT_DIR, "phase2_scored.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(audit_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "score", "semantic", "technical", "founding_fit", "behavioural", "career_quality"])
        for r in top:
            writer.writerow([r.candidate_id, round(r.score, 4),
                             round(r.component_scores.semantic, 4),
                             round(r.component_scores.technical, 4),
                             round(r.component_scores.founding_fit, 4),
                             round(r.component_scores.behavioural, 4),
                             round(r.component_scores.career_quality, 4)])
    print(f"  Audit: {audit_path} ({len(top)} rows)")

    return top
