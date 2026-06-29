#!/usr/bin/env python3
"""
Single entry point. Run: python run.py
"""

import argparse
import csv
import os
import time
from src.config import DATA_PATH, CALIBRATION_CACHE_PATH, JD_CACHE_PATH, OUTPUT_DIR
from src.phase0_calibrate import run_calibration, decompose_jd, JD_TEXT
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
    parser.add_argument("--use-reasoning-cache", action="store_true",
                        help="Use cached reasoning if available (default: always call API)")
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
        final = generate_reasoning_batch(final, candidates_map, jd, use_cache=args.use_reasoning_cache)
        print(f"  Done in {time.time()-t4:.1f}s")

    total = time.time() - t0
    print(f"\nTotal pipeline time: {total:.1f}s")

    if not args.dry_run:
        write_submission(final)
        print(f"Submission written to {OUTPUT_DIR}submission.csv")


def write_submission(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "submission.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for r in results:
            reasoning = r.reasoning.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
            writer.writerow({
                "candidate_id": r.candidate_id,
                "rank": r.rank,
                "score": round(r.score, 4),
                "reasoning": reasoning,
            })


if __name__ == "__main__":
    main()
