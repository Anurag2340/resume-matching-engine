# Redrob AI — Prompt Usage Documentation
## Redrob AI Campus Hackathon | Resume Matching Engine

---

## FORM SUBMISSION — Top 3 Key Prompts

### ✅ Prompt 1 (Most Important)
```
I have 10 student resumes with noisy/misspelled skill strings and a SKILL_ALIASES 
dictionary. For each resume: split raw skills on commas, lowercase every token, 
match multi-word phrases (like "spring boot", "feature engineering", "competitive 
programming") BEFORE single tokens, apply SKILL_ALIASES mapping, discard any token 
not in the alias map, then deduplicate canonical skills preserving first-seen order. 
Show me the normalized skill list for every resume.
```

### ✅ Prompt 2
```
Using the normalized resume skills, build a shared vocabulary sorted alphabetically. 
Then compute TF-IDF vectors for all 10 resumes using exactly:
  TF = 1/N  (N = total unique skills in that resume after deduplication)
  IDF = ln(10 / df(skill))  using natural logarithm, no smoothing
  TF-IDF = TF × IDF
Show me each resume's TF-IDF values and the Euclidean norm of each vector. 
Use no external libraries — only Python's math module.
```

### ✅ Prompt 3
```
For each of the 3 JDs, normalize required + preferred skills using the same 
SKILL_ALIASES, build a binary vector (1 if skill in vocabulary, 0 otherwise). 
Then compute cosine similarity: Cosine(A,B) = (A·B)/(|A|×|B|) between every 
resume TF-IDF vector and each JD binary vector. Rank Top 3 candidates per JD 
by score descending, break ties alphabetically. Round scores to 2 decimal places.
```

---

## Full Staged Prompt Workflow (All Steps)

### Stage 1 — Understand the Problem
```
Prompt 1:
Read this problem: I need to build a Resume Matching Engine.
I have 10 resumes with noisy skill strings and 3 job descriptions.
Break the problem into clear stages I should implement one by one.
```

### Stage 2 — Explore Noisy Tokens
```
Prompt 2:
Here are the 10 raw resume skill strings:
"Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"
"JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"
"Java, Spring Boot, MySql, Microservices, Docker, kubernates"
"Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"
"C++, Algoritms, Data Structure, competitive programming, python"
"javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"
"Python, Sklearn, XGboost, feature engineering, SQL, tableau"
"Java, Android, Kotlin, Firebase, REST, UI/UX, figma"
"Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"
"python, R, statistics, ML, regression, clustering, Power-BI"

List all unique raw tokens and identify which ones are misspelled 
or use non-standard formatting.
```

### Stage 3 — Build Normalization Logic
```
Prompt 3:
Using this exact SKILL_ALIASES dictionary [paste full dict], write a Python 
function normalize_skills(raw_string) that:
1. Splits on commas
2. Lowercases each token
3. Tries to match multi-word phrase aliases FIRST (sorted longest to shortest)
4. Then tries single-token aliases
5. Discards tokens not in SKILL_ALIASES
6. Deduplicates canonical skills (first-seen order)
Returns a list of canonical skill strings.
Use only Python standard library.
```

### Stage 4 — Validate Normalization
```
Prompt 4:
Run normalize_skills() on all 10 resumes and print the canonical skill 
list for each candidate. Specifically verify these tricky cases:
- "MachineLearning" → should be "machine_learning"
- "Sklearn" → should be "machine_learning" (same canonical)
- "data-viz" AND "matplotlib" → both → "data_visualization" → should deduplicate to 1
- "Spring Boot" → "spring_boot" (multi-word, must be matched before "spring" or "boot")
- "feature engineering" → "feature_engineering" (two-word)
- "kubernates" → "kubernetes" (typo)
Show me N (unique skill count) for each resume.
```

### Stage 5 — Build Vocabulary
```
Prompt 5:
From the normalized + deduplicated resume skills of all 10 resumes,
build a shared vocabulary. Sort it alphabetically. Print all terms 
with their index. This vocabulary will be used for both resume TF-IDF 
vectors and JD binary vectors.
```

### Stage 6 — Compute Document Frequency
```
Prompt 6:
For each skill in the vocabulary, count df(skill) = number of resumes 
that contain it. Then compute IDF(skill) = ln(10 / df(skill)) using 
Python's math.log() (natural log). No smoothing. 
Print the df and IDF value for every vocabulary term.
Verify: python should have df=6 (appears in 6 resumes) and the 
lowest IDF since it's most common.
```

### Stage 7 — Compute TF-IDF Vectors
```
Prompt 7:
For each resume, compute the TF-IDF vector over the vocabulary:
  TF(skill, resume) = 1 / N   where N = number of unique skills in that resume
  IDF(skill) = ln(10 / df(skill))
  TF-IDF = TF × IDF
Vector length = vocabulary size (48). Non-matching skills = 0.
Also compute the Euclidean norm ||A|| = sqrt(sum of squares) for each vector.
Print TF-IDF values and ||A|| for each of the 10 resumes.
```

### Stage 8 — Build JD Binary Vectors
```
Prompt 8:
For each JD, combine required + preferred skills into one string. 
Normalize using normalize_skills() with the same SKILL_ALIASES.
Build a binary vector over the vocabulary: 1.0 if skill exists in vocab, 
0.0 otherwise.
Important: skills like "pytorch" and "redis" do NOT appear in any resume 
vocabulary — they must be silently ignored.
Print the mapped skills and ||B|| for each JD.
Expected:
  JD1: 10 skills in vocab → ||B|| = sqrt(10) = 3.1623
  JD2: 9 skills in vocab  → ||B|| = sqrt(9)  = 3.0000
  JD3: 11 skills in vocab → ||B|| = sqrt(11) = 3.3166
```

### Stage 9 — Cosine Similarity
```
Prompt 9:
Compute cosine similarity between each resume TF-IDF vector (A) and 
each JD binary vector (B):
  Cosine(A,B) = (A · B) / (||A|| × ||B||)
Print a full 10×3 matrix showing every candidate vs every JD score 
to 4 decimal places. Then for each JD, sort descending and show Top 3.
```

### Stage 10 — Final Output + Tie-breaking
```
Prompt 10:
Format the final output exactly as required:
  JD-1 — Kakao (ML Engineer)
  Name(score), Name(score), Name(score)
  
  JD-2 — Naver (Backend Engineer)
  Name(score), Name(score), Name(score)

  JD-3 — Line (Frontend Engineer)
  Name(score), Name(score), Name(score)

Scores rounded to 2 decimal places.
Ties broken alphabetically by candidate name.
```

### Stage 11 — Clean Code + Debug Mode
```
Prompt 11:
Refactor all the above into one clean Python file main.py with:
- Proper docstrings for every function
- Clear section comments (Step 1 through Step 6)
- A debug mode that prints normalized skills, vocabulary, 
  TF-IDF values, and the full similarity matrix
- Only math from standard library
- Follows the exact algorithm specification from the problem sheet
```

### Stage 12 — Edge Case Verification
```
Prompt 12:
Verify these specific edge cases in the final code:
1. "feature engineering" (two words separated by space) — does it match 
   multi-word alias correctly after comma-split?
2. "Sneha Patel" has both "data-viz" and "matplotlib" → both map to 
   data_visualization → after dedup N should be 6 not 7
3. "Karan Mehta" has "Sklearn" → maps to machine_learning, and also 
   NO explicit "machine_learning" → N=6
4. JD1 has "PyTorch" → normalizes to "pytorch" → NOT in resume vocab 
   → correctly ignored in JD vector
5. JD2 has "Redis" → normalizes to "redis" → NOT in vocab → ignored
Show me the output confirming each of these.
```

---

*All prompts used with Redrob AI only. No other AI tools were used.*
