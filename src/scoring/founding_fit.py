"""
Founding team fit scoring.

EDA Finding 1: Only 0.9% of candidates have AI titles, so founding_fit
must rely on company-size proxy and scope breadth, not title keywords.

Weights: startup=0.50, scope=0.25, velocity=0.25
Startup detection uses company_size field (< 50 employees) rather than
name-pattern exclusion, which was falsely inflating scores for candidates
at unknown-but-established companies.
"""

from src.config import STARTUP_EMPLOYEE_THRESHOLD, MIN_SCOPE_BREADTH_FOR_BONUS, FAST_VELOCITY_YEARS

LARGE_COMPANY_PATTERNS = [
    # Global tech
    "google", "microsoft", "amazon", "meta", "apple", "netflix", "uber",
    "airbnb", "salesforce", "oracle", "ibm", "adobe", "cisco", "intel",
    "zoom", "twitter", "linkedin", "spotify", "nvidia", "vmware",
    # Indian tech
    "flipkart", "ola", "zomato", "swiggy", "paytm", "phonepe", "cred",
    "meesho", "nykaa", "razorpay", "unacademy", "zoho", "freshworks",
    "inmobi", "byju", "vedantu", "upgrad", "glance", "dream11",
    "pharmeasy", "policybazaar", "genpact",
    # Consulting / services
    "accenture", "deloitte", "infosys", "wipro", "tcs", "tata consultancy",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "persistent systems",
    # Banking / finance
    "goldman sachs", "jpmorgan", "morgan stanley", "barclays", "credit suisse",
    "deutsche bank", "hsbc", "wells fargo",
]

DOMAIN_KEYWORDS = {
    "ml/ai": ["machine learning", "deep learning", "nlp", "ai", "llm", "neural", "embedding", "pytorch", "tensorflow"],
    "backend/infra": ["backend", "api", "microservice", "pipeline", "distributed", "database", "sql", "cache", "queue"],
    "data engineering": ["etl", "data pipeline", "spark", "airflow", "warehouse", "dbt", "bigquery", "snowflake"],
    "research": ["research", "publication", "experiment", "evaluation", "benchmark", "paper", "novel"],
    "leadership": ["lead", "manage", "team lead", "head of", "director", "mentor", "own"],
    "product/strategy": ["product", "stakeholder", "roadmap", "strategy", "okr", "kpi", "cross-functional"],
    "frontend": ["react", "angular", "vue", "frontend", "ui", "css", "javascript", "typescript"],
    "devops/cloud": ["aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd", "deploy"],
}


def is_startup_role(role):
    """Returns True if role is at a startup (< 50 employees and not a known large company)."""
    size = role.raw.get("company_size", "")
    if size in ("1-10", "11-50"):
        # Small company — likely a startup unless it's a known large brand
        name = role.company.lower().strip()
        for pat in LARGE_COMPANY_PATTERNS:
            if pat in name:
                return False
        return True
    return False


def compute_scope_breadth(role, all_skills):
    """Count distinct high-level skill domains present in role description and title."""
    text = ((role.description or "") + " " + (role.title or "")).lower()
    for skill in all_skills:
        text += " " + (skill.name or "")
    count = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                count += 1
                break
    return count


def compute_seniority_velocity(candidate):
    """
    Find year of first Senior/Lead/Principal/Staff/Director title.
    velocity_years = year_of_first_senior - estimated_graduation_year
    """
    grad_year = None
    for edu in candidate.education:
        ey = edu.get("end_year")
        if ey and (grad_year is None or ey < grad_year):
            grad_year = ey
    if grad_year is None:
        grad_year = 2016  # fallback for synthetic data

    first_senior_year = None
    for role in candidate.career:
        title = (role.title or "").lower()
        if any(kw in title for kw in ["senior", "lead", "principal", "staff", "director", "head"]):
            raw = role.raw
            sd = raw.get("start_date", "")
            if sd and len(sd) >= 4:
                try:
                    y = int(sd[:4])
                    if first_senior_year is None or y < first_senior_year:
                        first_senior_year = y
                except (ValueError, TypeError):
                    pass

    if first_senior_year is None:
        return 0.5

    velocity = first_senior_year - grad_year
    if velocity <= FAST_VELOCITY_YEARS:
        return 1.0
    elif velocity <= FAST_VELOCITY_YEARS * 2:
        return 0.6
    else:
        return 0.3


def compute_founding_fit_score(candidate):
    """
    Combines three sub-scores:
    1. startup_score (0.50): ratio of roles at genuinely small companies (< 50 employees)
    2. scope_score (0.25): mean scope breadth across roles
    3. velocity_score (0.25): seniority velocity
    """
    career = candidate.career
    if not career:
        return 0.0

    n_roles = len(career)

    # Startup score — count roles at genuinely small companies (< 50 employees)
    startup_roles = 0
    for role in career:
        if role.is_startup or is_startup_role(role):
            startup_roles += 1
    startup_score = startup_roles / n_roles if n_roles > 0 else 0.0

    # Scope score
    breadths = [compute_scope_breadth(role, candidate.skills) for role in career]
    mean_breadth = sum(breadths) / len(breadths) if breadths else 0
    scope_score = min(mean_breadth / 8.0, 1.0)

    # Bonus for high scope breadth
    if any(b >= MIN_SCOPE_BREADTH_FOR_BONUS for b in breadths):
        scope_score = min(scope_score + 0.15, 1.0)

    # Velocity score
    velocity_score = compute_seniority_velocity(candidate)

    return max(0.0, min(1.0, 0.50 * startup_score + 0.25 * scope_score + 0.25 * velocity_score))
