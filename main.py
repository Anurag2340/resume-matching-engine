"""
========================================================
  Resume Matching Engine
  Redrob AI Campus Hackathon — Individual Competition
  Powered by McKinley Rice
========================================================

Pipeline:
  Step 1 → Normalize raw resume skills via SKILL_ALIASES
  Step 2 → Deduplicate canonical skills per resume
  Step 3 → Build shared alphabetical vocabulary
  Step 4 → Compute TF-IDF vectors for resumes
  Step 5 → Build binary vectors for JDs
  Step 6 → Cosine similarity → Top 3 per JD

Formulas:
  TF(skill, resume) = 1 / N          [N = unique skills after dedup]
  IDF(skill)        = ln(10 / df)    [natural log, no smoothing]
  TF-IDF            = TF × IDF
  Cosine(A,B)       = (A·B) / (|A|×|B|)

Rules:
  - No external libraries (only built-in math)
  - SKILL_ALIASES used exactly as provided
  - Multi-word phrases matched before single tokens
  - Ties broken alphabetically by candidate name
"""

import math


# ─────────────────────────────────────────────────────────────────
#  SKILL ALIASES  (use exactly as provided — do NOT modify)
# ─────────────────────────────────────────────────────────────────

SKILL_ALIASES = {
    # Languages
    "python": "python",         "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp",               "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data Science
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",   "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",

    # Web — Frontend
    "react": "react",     "reacts": "react",    "reactjs": "react",
    "vue": "vue",         "vue.js": "vue",       "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",  "html css": "html_css",
    "html": "html_css",      "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",

    # Web — Backend
    "node.js": "nodejs",  "nodejs": "nodejs",   "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",  "springboot": "spring_boot",
    "rest api": "rest_api",        "rest": "rest_api",  "restapi": "rest_api",
    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",     "mysq": "mysql",
    "postgresql": "postgresql",  "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",  "kubernates": "kubernetes",  "k8s": "kubernetes",
    "ci/cd": "ci_cd",   "cicd": "ci_cd",   "ci cd": "ci_cd",
    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS Fundamentals
    "algorithms": "algorithms",   "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",   "ui ux": "ui_ux",
    "figma": "figma",
}

# Pre-sort multi-word keys longest-first → matched before single tokens
_MULTI_WORD_KEYS = sorted(
    [k for k in SKILL_ALIASES if " " in k],
    key=lambda x: -len(x)
)


# ─────────────────────────────────────────────────────────────────
#  RAW DATASETS
# ─────────────────────────────────────────────────────────────────

RESUMES = [
    {"id": "01", "name": "Arjun Sharma",
     "raw": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "name": "Priya Nair",
     "raw": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "name": "Rahul Gupta",
     "raw": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "name": "Sneha Patel",
     "raw": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "name": "Vikram Singh",
     "raw": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "name": "Ananya Krishnan",
     "raw": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "name": "Karan Mehta",
     "raw": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "name": "Deepika Rao",
     "raw": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "name": "Aditya Kumar",
     "raw": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "name": "Meera Iyer",
     "raw": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

JDS = [
    {
        "id": "JD1",
        "label": "JD-1 — Kakao (ML Engineer)",
        "required": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
        "preferred": "NLP, BERT, Feature Engineering, Statistics",
    },
    {
        "id": "JD2",
        "label": "JD-2 — Naver (Backend Engineer)",
        "required": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
        "preferred": "REST API, CI/CD, Redis",
    },
    {
        "id": "JD3",
        "label": "JD-3 — Line (Frontend Engineer)",
        "required": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "preferred": "Node.js, GraphQL, Redux, Jest, AWS",
    },
]


# ─────────────────────────────────────────────────────────────────
#  STEP 1 & 2 — NORMALIZE + DEDUPLICATE
# ─────────────────────────────────────────────────────────────────

def normalize_skills(raw: str) -> list:
    """
    Split raw skill string on commas → lowercase each token →
    match multi-word aliases first → single-token aliases next →
    discard unknowns → deduplicate (preserve first-seen order).
    """
    tokens = [t.strip().lower() for t in raw.split(",")]
    seen, canonical = set(), []

    for token in tokens:
        matched = False

        # Try multi-word phrase aliases first (longest → shortest)
        for phrase in _MULTI_WORD_KEYS:
            if token == phrase:
                c = SKILL_ALIASES[phrase]
                if c not in seen:
                    seen.add(c)
                    canonical.append(c)
                matched = True
                break

        # Single-token alias
        if not matched and token in SKILL_ALIASES:
            c = SKILL_ALIASES[token]
            if c not in seen:
                seen.add(c)
                canonical.append(c)
        # else: unknown token → silently discarded

    return canonical


# ─────────────────────────────────────────────────────────────────
#  STEP 3 — VOCABULARY
# ─────────────────────────────────────────────────────────────────

def build_vocabulary(resumes: list) -> list:
    """Sorted alphabetical union of all resume canonical skills."""
    return sorted(set(s for r in resumes for s in r["skills"]))


# ─────────────────────────────────────────────────────────────────
#  STEP 4 — TF-IDF
# ─────────────────────────────────────────────────────────────────

def compute_tfidf(resumes: list, vocab: list) -> None:
    """
    TF  = 1/N  (each unique skill contributes equally after dedup)
    IDF = ln(total_docs / df)  — natural log, no smoothing
    Stores 'tfidf' vector (len = |vocab|) into each resume dict.
    """
    word_idx  = {w: i for i, w in enumerate(vocab)}
    total_docs = len(resumes)  # 10

    # Document frequency per skill
    df = {skill: sum(1 for r in resumes if skill in r["skills"])
          for skill in vocab}

    # IDF per skill
    idf = {skill: math.log(total_docs / df[skill]) for skill in vocab}

    # Build TF-IDF vector for each resume
    for r in resumes:
        n   = len(r["skills"])          # unique skill count (post-dedup)
        vec = [0.0] * len(vocab)
        for skill in r["skills"]:
            vec[word_idx[skill]] = (1.0 / n) * idf[skill]
        r["tfidf"] = vec


# ─────────────────────────────────────────────────────────────────
#  STEP 5 — JD BINARY VECTORS
# ─────────────────────────────────────────────────────────────────

def build_jd_vector(jd: dict, vocab: list) -> list:
    """
    Normalize JD required + preferred skills → map to vocab indices → 
    binary vector (1.0 if skill present in vocab, else ignored).
    """
    word_idx = {w: i for i, w in enumerate(vocab)}
    combined = jd["required"] + ", " + jd["preferred"]
    skills   = normalize_skills(combined)

    vec = [0.0] * len(vocab)
    for skill in skills:
        if skill in word_idx:
            vec[word_idx[skill]] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────
#  STEP 6 — COSINE SIMILARITY & RANKING
# ─────────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    """Cosine(A, B) = (A·B) / (|A| × |B|)"""
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_candidates(resumes: list, jd_vec: list, top_n: int = 3) -> list:
    """Sort by cosine score descending; ties broken alphabetically by name."""
    scores = [(r["name"], cosine_similarity(r["tfidf"], jd_vec))
              for r in resumes]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_n]


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # ── Step 1 & 2: Normalize + Deduplicate ─────────────────────
    for r in RESUMES:
        r["skills"] = normalize_skills(r["raw"])

    # ── Step 3: Vocabulary ───────────────────────────────────────
    vocab = build_vocabulary(RESUMES)

    # ── Step 4: TF-IDF ──────────────────────────────────────────
    compute_tfidf(RESUMES, vocab)

    # ── Steps 5 & 6: JD Vectors + Ranking ───────────────────────
    print("=" * 60)
    print("  RESUME MATCHING ENGINE — FINAL RESULTS")
    print("=" * 60)

    for jd in JDS:
        jd_vec = build_jd_vector(jd, vocab)
        top3   = rank_candidates(RESUMES, jd_vec)
        print(f"\n{jd['label']}")
        print(", ".join(f"{name}({score:.2f})" for name, score in top3))

    print("\n" + "=" * 60)

    # ── Debug: Intermediate Values ───────────────────────────────
    print("\n[DEBUG] Normalized + Deduplicated Skills:")
    for r in RESUMES:
        print(f"  {r['name']:20s} (N={len(r['skills'])}): {r['skills']}")

    print(f"\n[DEBUG] Vocabulary → {len(vocab)} terms:")
    for i, term in enumerate(vocab):
        print(f"  [{i:02d}] {term}")

    print("\n[DEBUG] Full Cosine Similarity Matrix:")
    header = f"  {'Candidate':20s}" + "".join(f"  {jd['id']:>6}" for jd in JDS)
    print(header)
    for r in RESUMES:
        row = f"  {r['name']:20s}"
        for jd in JDS:
            jd_vec = build_jd_vector(jd, vocab)
            s = cosine_similarity(r["tfidf"], jd_vec)
            row += f"  {s:6.4f}"
        print(row)


if __name__ == "__main__":
    main()
