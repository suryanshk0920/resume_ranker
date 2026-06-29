"""
Technical depth scoring.

Per EDA Finding 5: skill durations legitimately overlap across jobs.
Skills with 0 duration AND 0 endorsements score 0.0 (keyword stuffers).
All other skills are scored by relevance x proficiency x duration x endorsements.
Endorsements and duration are individual quality signals, not honeypot flags.
"""

import math

# AI/ML skill keywords — case-insensitive substring match
AI_SKILL_KEYWORDS = [
    "python", "pytorch", "tensorflow", "transformers", "huggingface",
    "llm", "nlp", "embeddings", "vector", "rag", "retrieval", "ranking",
    "recommendation", "deep learning", "neural", "bert", "gpt", "fine-tuning",
    "mlops", "kubernetes", "docker", "fastapi", "sql", "spark", "distributed",
    "reinforcement learning", "computer vision", "ocr", "speech", "multimodal",
    "langchain", "pinecone", "faiss", "weaviate", "elasticsearch",
]

# General engineering keywords
GENERAL_ENG_KEYWORDS = [
    "git", "linux", "agile", "scrum", "jira", "rest", "api", "ci/cd",
    "unit testing", "bash", "vscode", "postman", "swagger",
]


def skill_relevance_weight(skill_name):
    """Return relevance weight based on skill name matching."""
    name = skill_name.lower()
    for kw in AI_SKILL_KEYWORDS:
        if kw in name:
            return 1.0
    for kw in GENERAL_ENG_KEYWORDS:
        if kw in name:
            return 0.3
    return 0.1


def compute_technical_score(candidate, calibration):
    """
    For each skill:
      1. relevance = skill_relevance_weight(skill.name)
      2. prof_weight = skill.proficiency  # already normalised 0.0-1.0
      3. dur_ceil = min(skill.duration_months / 36.0, 1.0)
      4. end_weight = log(skill.endorsements + 1) / log(endorsement_p90 + 1)
         clamped to [0.0, 1.0]
      5. skill_score = relevance * min(prof_weight, dur_ceil) * (0.6 + 0.4 * end_weight)

    Zero-duration + zero-endorsement skills: skill_score = 0.0 always.

    candidate_technical_score = mean(top 10 skill scores by individual value)
    """
    endorsement_p90 = calibration.get("endorsement_p90", 50)

    skill_scores = []
    for skill in candidate.skills:
        dur = skill.duration_months
        endorse = skill.endorsements

        # Zero-duration + zero-endorsement = keyword stuffer
        if dur == 0 and endorse == 0:
            skill_scores.append(0.0)
            continue

        relevance = skill_relevance_weight(skill.name)
        prof = skill.proficiency
        dur_ceil = min(dur / 36.0, 1.0)
        end_norm = math.log(endorse + 1) / max(math.log(endorsement_p90 + 1), 1e-6)
        end_weight = max(0.0, min(1.0, end_norm))

        skill_score = relevance * min(prof, dur_ceil) * (0.6 + 0.4 * end_weight)
        skill_scores.append(skill_score)

    if not skill_scores:
        return 0.0

    # Top 10 by score
    top10 = sorted(skill_scores, reverse=True)[:10]
    return sum(top10) / len(top10)
