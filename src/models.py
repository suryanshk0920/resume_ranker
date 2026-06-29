from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CareerRole:
    title: str
    company: str
    duration_months: int
    description: str
    is_startup: bool = False
    scope_breadth: int = 0
    raw: dict = field(default_factory=dict)


@dataclass
class Skill:
    name: str
    proficiency: float
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
    raw: dict = field(default_factory=dict)


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
    gate_fail_reason: str = ""
    company: str = ""
