import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDINGS_CACHE_PATH

_MODEL = None

NEGATIVE_JD_TEXT = (
    "Marketing manager, sales executive, business analyst, HR manager, "
    "project coordinator with no technical background. Managed stakeholders, "
    "handled client accounts, coordinated project timelines. No machine "
    "learning, no model development, no AI engineering experience. "
    "Excel, PowerPoint, CRM tools, cold calling, lead generation."
)


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def build_jd_embedding(jd_decomposed):
    cache_path = EMBEDDINGS_CACHE_PATH + "_jd.pkl"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    model = _get_model()

    # Positive JD embedding — use full JD text for richer signal
    from src.phase0_calibrate import JD_TEXT
    jd_text = JD_TEXT
    pos_emb = model.encode(jd_text, normalize_embeddings=True)

    # Negative JD embedding
    neg_emb = model.encode(NEGATIVE_JD_TEXT, normalize_embeddings=True)

    result = {"positive": pos_emb, "negative": neg_emb}

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def build_candidate_embeddings(candidates):
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    import torch
    torch.set_num_threads(4)

    model = _get_model()
    recency_weights = [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05]

    all_texts = []
    index_map = []

    for ci, cand in enumerate(candidates):
        for ri, role in enumerate(cand.career[:5]):
            desc = (role.description or "")[:512]
            text = f"{role.title} at {role.company}: {desc}"
            all_texts.append(text)
            w = recency_weights[ri] if ri < len(recency_weights) else 0.05
            index_map.append((ci, ri, w))
        if cand.headline:
            all_texts.append(cand.headline)
            index_map.append((ci, -1, 0.5))
        # Include skills as a single text block
        if cand.skills:
            skill_text = "Skills: " + ", ".join(s.name for s in cand.skills[:20])
            all_texts.append(skill_text)
            index_map.append((ci, -2, 0.3))
        # Include summary
        if cand.summary:
            all_texts.append(cand.summary[:512])
            index_map.append((ci, -3, 0.4))

    embeddings = model.encode(
        all_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    candidate_embeddings = {}
    agg_map = {}
    for (ci, ri, w), emb in zip(index_map, embeddings):
        if ci not in agg_map:
            agg_map[ci] = []
        agg_map[ci].append((w, emb))

    for ci, weighted_embs in agg_map.items():
        total_w = sum(w for w, _ in weighted_embs)
        if total_w > 0:
            avg = sum(w * emb for w, emb in weighted_embs) / total_w
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            candidate_embeddings[candidates[ci].candidate_id] = avg

    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH) or ".", exist_ok=True)
    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump(candidate_embeddings, f)

    return candidate_embeddings


def compute_semantic_scores(candidates, jd_embeddings, candidate_embeddings):
    """
    Contrastive scoring: positive_sim - 0.5 * negative_sim.
    Then min-max normalise across the pool to use full 0-1 range.
    """
    pos_emb = jd_embeddings["positive"]
    neg_emb = jd_embeddings["negative"]

    raw = {}
    for cand in candidates:
        emb = candidate_embeddings.get(cand.candidate_id)
        if emb is not None:
            pos_sim = float(np.dot(emb, pos_emb))
            neg_sim = float(np.dot(emb, neg_emb))
            raw[cand.candidate_id] = pos_sim - 0.4 * neg_sim
        else:
            raw[cand.candidate_id] = 0.0

    vals = list(raw.values())
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng > 1e-12:
        scores = {cid: (v - mn) / rng for cid, v in raw.items()}
    else:
        scores = {cid: 0.5 for cid in raw}

    return scores
