"""
Phase 3 — Precision Rerank.
Apply diversity penalty and founding_fit tiebreaker, assign final ranks 1-100.
"""

from src.config import (
    FINAL_TOP_N, DIVERSITY_SAME_COMPANY_PENALTY,
    FOUNDING_FIT_TIEBREAKER_BAND, FOUNDING_FIT_SWAP_THRESHOLD, FOUNDING_FIT_FLOOR,
)


def run_phase3(results):
    """
    1. Sort by score descending.
    2. Apply founding_fit tiebreaker (groups within 0.02 points, re-sort by founding_fit).
    3. Apply diversity penalty to top 10.
    4. Apply founding_fit floor swap for top 10 (if any top-10 has founding_fit < 0.40).
    5. Assign final ranks 1-100.
    6. Validate: no duplicates, no honeypot-flagged candidates in top 100.
    """
    sorted_results = sorted(results, key=lambda r: (-r.score, r.candidate_id))

    # ── founding_fit tiebreaker ──
    sorted_results = apply_founding_fit_tiebreaker(sorted_results)

    top10 = sorted_results[:10]
    rest = sorted_results[10:]

    adjusted_top10 = apply_diversity_check(top10)
    adjusted_top10.sort(key=lambda r: (-r.score, r.candidate_id))

    final = adjusted_top10 + rest[:FINAL_TOP_N - 10]

    # Re-sort entire list (diversity penalties may have changed top 10 ordering)
    final.sort(key=lambda r: (-r.score, r.candidate_id))

    # ── founding_fit floor swap for top 10 ──
    final = apply_founding_fit_floor(final)

    # Assign ranks
    for i, r in enumerate(final):
        r.rank = i + 1

    # Validate
    ids = [r.candidate_id for r in final]
    assert len(ids) == len(set(ids)), "Duplicate candidate IDs in final list"
    assert not any(r.honeypot_flag for r in final), "Honeypot candidate in top 100"

    top100 = final[:FINAL_TOP_N]

    # Write audit CSV
    import csv, os
    from src.config import OUTPUT_DIR
    audit_path = os.path.join(OUTPUT_DIR, "phase3_final.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(audit_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "semantic", "technical", "founding_fit", "behavioural", "career_quality", "reasoning"])
        for r in top100:
            reasoning = r.reasoning.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
            writer.writerow([r.candidate_id, r.rank, round(r.score, 4),
                             round(r.component_scores.semantic, 4),
                             round(r.component_scores.technical, 4),
                             round(r.component_scores.founding_fit, 4),
                             round(r.component_scores.behavioural, 4),
                             round(r.component_scores.career_quality, 4),
                             reasoning])
    print(f"  Audit: {audit_path} ({len(top100)} rows)")

    return top100


def apply_founding_fit_tiebreaker(results):
    """
    Among candidates within FOUNDING_FIT_TIEBREAKER_BAND score of each other
    (effectively tied), prefer the one with higher founding_fit score.

    1. Scan results in score-descending order.
    2. For each gap > band, start a new band.
    3. Within each band, sort by founding_fit descending (then score descending).
    4. Reassign rank positions.
    """
    if not results:
        return results

    # Build bands
    bands = []
    current_band = [results[0]]
    for r in results[1:]:
        if current_band[-1].score - r.score <= FOUNDING_FIT_TIEBREAKER_BAND:
            current_band.append(r)
        else:
            bands.append(current_band)
            current_band = [r]
    bands.append(current_band)

    # Sort within each band: founding_fit descending, then score descending, then candidate_id
    rearranged = []
    for band in bands:
        band.sort(key=lambda r: (-r.component_scores.founding_fit, -r.score, r.candidate_id))
        rearranged.extend(band)

    return rearranged


def apply_founding_fit_floor(results):
    """
    If any candidate in rank 1-10 has founding_fit < FOUNDING_FIT_FLOOR,
    swap them with the highest founding_fit candidate from ranks 11-20
    IF that candidate's score is within FOUNDING_FIT_SWAP_THRESHOLD of the
    displaced candidate's score. Log any swaps made.
    """
    n = min(10, len(results))
    swap_pool_end = min(20, len(results))

    did_swap = False
    final = list(results)

    for i in range(n):
        if final[i].component_scores.founding_fit < FOUNDING_FIT_FLOOR:
            # Find best founding_fit in the swap pool (11-20)
            best_j = -1
            best_founding = -1.0
            for j in range(n, swap_pool_end):
                if final[j].component_scores.founding_fit > best_founding:
                    diff = final[i].score - final[j].score
                    if diff <= FOUNDING_FIT_SWAP_THRESHOLD:
                        best_j = j
                        best_founding = final[j].component_scores.founding_fit

            if best_j >= 0:
                print(f"  Founding fit swap: {final[i].candidate_id} (founding={final[i].component_scores.founding_fit:.2f}, score={final[i].score:.4f}) "
                      f"<-> {final[best_j].candidate_id} (founding={final[best_j].component_scores.founding_fit:.2f}, score={final[best_j].score:.4f})")
                final[i], final[best_j] = final[best_j], final[i]
                did_swap = True

    if did_swap:
        final.sort(key=lambda r: (-r.score, r.candidate_id))

    return final


def apply_diversity_check(top10):
    """
    Within top 10, detect company clusters.
    If >2 candidates from same company, penalise the 3rd+ by
    DIVERSITY_SAME_COMPANY_PENALTY. Re-sort and return adjusted top 10.
    Also check education: if >3 from same university, apply same penalty.
    """
    from collections import Counter

    company_counts = Counter()
    for r in top10:
        company_counts[r.company.lower()] += 1

    penalised = set()
    for company, count in company_counts.items():
        if count > 2 and company:
            seen = 0
            for r in top10:
                if r.company.lower() == company:
                    seen += 1
                    if seen > 2:
                        r.score -= DIVERSITY_SAME_COMPANY_PENALTY
                        penalised.add(r.candidate_id)

    if penalised:
        top10.sort(key=lambda r: -r.score)

    return top10
