# 🎯 Resume Matching Engine
### Redrob AI Campus Hackathon — Individual Competition | Powered by McKinley Rice

---

## 📌 Problem Summary

Match **10 student resumes** against **3 Korean tech company JDs** using TF-IDF cosine similarity — built from scratch with zero external libraries.

---

## ⚙️ Pipeline

```
Raw Skills
    │
    ▼
Step 1 ── Normalize  → split on comma → lowercase → apply SKILL_ALIASES (multi-word first) → discard unknowns
    │
    ▼
Step 2 ── Deduplicate → each canonical skill appears only once per resume
    │
    ▼
Step 3 ── Vocabulary → sorted alphabetical union of all resume skills (48 terms)
    │
    ▼
Step 4 ── TF-IDF      → TF = 1/N · IDF = ln(10/df)  [no smoothing, natural log]
    │
    ▼
Step 5 ── JD Vectors  → binary vector over same vocabulary (required + preferred)
    │
    ▼
Step 6 ── Cosine Sim  → rank Top 3 per JD (ties → alphabetical)
```

---

## 📐 Formulas

| Formula | Expression |
|---------|-----------|
| **TF** | `1 / N` where N = unique skills in resume |
| **IDF** | `ln(10 / df(skill))` — natural log, no smoothing |
| **TF-IDF** | `TF × IDF` |
| **Cosine** | `(A · B) / (\|A\| × \|B\|)` |

---

## 🚀 Run

```bash
python main.py
```

> **No pip install needed** — uses only Python's built-in `math` module.

---

## ✅ Results

```
============================================================
  RESUME MATCHING ENGINE — FINAL RESULTS
============================================================

JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## 🔍 Key Implementation Notes

| Edge Case | Handling |
|-----------|---------|
| `Pyhton` → `python` | Typo alias in SKILL_ALIASES |
| `MachineLearning` → `machine_learning` | CamelCase without space |
| `Sklearn` → `machine_learning` | Same canonical as ML |
| `matplotlib` + `data-viz` → `data_visualization` | Both map to same → dedup keeps only 1 |
| `kubernates` → `kubernetes` | Spelling error alias |
| `Spring Boot` (two words) → `spring_boot` | Multi-word matched before single token |
| `pytorch`, `redis` not in vocab | No resume has them → correctly excluded from JD vectors |
| `python` df=6 → IDF=0.5108 | Most common → lowest weight |

---

## 📁 Vocabulary (48 terms)

```
algorithms, android, aws, bert, ci_cd, clustering, competitive_programming,
cpp, data_structures, data_visualization, deep_learning, docker,
feature_engineering, figma, firebase, flask, graphql, html_css, java,
javascript, jest, keras, kotlin, kubernetes, machine_learning, microservices,
mongodb, mysql, nlp, nodejs, numpy, pandas, postgresql, python, r, react,
redux, regression, rest_api, spring_boot, sql, statistics, tailwind,
tensorflow, typescript, ui_ux, vue, xgboost
```

---

## 📊 Full Similarity Matrix

| Candidate | JD-1 | JD-2 | JD-3 |
|-----------|------|------|------|
| Arjun Sharma | **0.40** | 0.00 | 0.00 |
| Priya Nair | 0.00 | 0.12 | **0.58** |
| Rahul Gupta | 0.00 | **0.81** | 0.00 |
| Sneha Patel | **0.57** | 0.00 | 0.00 |
| Vikram Singh | 0.03 | 0.00 | 0.00 |
| Ananya Krishnan | 0.03 | **0.28** | **0.35** |
| Karan Mehta | **0.53** | 0.00 | 0.00 |
| Deepika Rao | 0.00 | **0.19** | 0.09 |
| Aditya Kumar | 0.00 | 0.00 | **0.67** |
| Meera Iyer | 0.33 | 0.00 | 0.00 |

---

## 🛡️ Rules Followed

- ✅ Only `math` standard library used
- ✅ `SKILL_ALIASES` used exactly as provided, not modified
- ✅ Multi-word phrases matched before single tokens
- ✅ No TF-IDF computed for JDs (binary vectors only)
- ✅ Vocabulary built from resume skills only
- ✅ Ties broken alphabetically by candidate name
- ✅ Natural logarithm used — no smoothing

---

*Redrob AI Campus Hackathon · Powered by McKinley Rice*
