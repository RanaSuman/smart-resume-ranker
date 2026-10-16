# 📄 Smart Resume Screening & Ranking System

> **SDG 8: Decent Work and Economic Growth** — fair, objective, bias-reduced
> shortlisting that helps every candidate get judged on skills, not formatting.

An ATS-style assistant that compares a **Job Description (JD)** against
**multiple resumes (PDF/DOCX)** using a classic, explainable IR pipeline:
**text extraction → spaCy lemmatization → TF-IDF → cosine similarity**,
served through an interactive **Streamlit** dashboard with rankings,
score distributions, and per-candidate **skill-gap analysis**.

---

## ✨ Features

- 📤 Batch upload of **50+ resumes** (PDF / DOCX) + JD (PDF / DOCX / TXT / paste)
- 🧹 spaCy `en_core_web_sm` pipeline: PII redaction, normalization, lemmatization
- 🧮 Sparse **TF-IDF (1–2 grams, sublinear TF)** + **cosine similarity** → 0–100% match
- 🏆 Ranked table, bar chart, score-distribution histogram
- 🔍 Skill gaps: top JD keywords missing from each resume
- ⬇️ One-click CSV export of rankings
- 🔒 Privacy-first: **in-memory only** (`io.BytesIO`), filename sanitization,
  type whitelist, 10 MB/file cap, optional PII redaction, zero disk persistence

---

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  JD +       │     │  parser.py       │     │  preprocessor.py    │
│  Resumes    │────▶│  sanitize name   │────▶│  redact PII         │
│  (PDF/DOCX) │     │  validate type   │     │  normalize + spaCy  │
│  in-memory  │     │  extract text    │     │  lemmatize (pipe)   │
└─────────────┘     └──────────────────┘     └────────┬────────────┘
                                                     │
                                            ┌────────▼────────────┐
                                            │  matcher.py         │
                                            │  TF-IDF sparse mat  │
                                            │  cosine similarity  │
                                            │  rank + skill gaps  │
                                            └────────┬────────────┘
                                                     │
                                            ┌────────▼────────────┐
                                            │  app.py (Streamlit) │
                                            │  rankings · charts  │
                                            │  gaps · CSV export  │
                                            └─────────────────────┘
```

| Stage | Module | Complexity |
|---|---|---|
| Extraction | `utils/parser.py` | O(P) pages/paragraphs, O(D) text memory |
| Preprocessing | `utils/preprocessor.py` | O(total tokens), batched `nlp.pipe` |
| Vectorize + score | `utils/matcher.py` | **O(N·M)** sparse; space **O(nnz)** |
| Rendering | `app.py` | O(R) candidates |

---

## 📁 Project Structure

```
smart-resume-ranker/
├── app.py                # Main Streamlit web application
├── requirements.txt      # Pinned dependency versions
├── README.md             # This file
├── utils/
│   ├── __init__.py
│   ├── parser.py         # PDF/DOCX text extraction (secure, in-memory)
│   ├── preprocessor.py   # Cleaning, PII redaction, spaCy lemmatization
│   └── matcher.py        # TF-IDF vectorization + cosine similarity engine
└── tests/
    ├── __init__.py
    └── test_pipeline.py  # pytest: parser, preprocessor, matcher
```

---

## 🚀 Installation & Usage

**Prerequisites:** Python 3.10+

```bash
# 1. Clone / enter the project
cd smart-resume-ranker

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the spaCy language model (if not pulled via requirements)
python -m spacy download en_core_web_sm

# 5. Run the app
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`):

1. Upload or paste the **Job Description**.
2. Upload candidate **resumes** (multiple PDF/DOCX allowed).
3. Tune settings in the sidebar (PII redaction, vocab size, gaps, score filter).
4. Click **🚀 Rank Candidates** → review rankings, charts, skill gaps.
5. Download results via **⬇️ Download rankings (CSV)**.

---

## ⚡ Performance Benchmarks

| Metric | Design | Typical result |
|---|---|---|
| Vectorize + score (50 resumes) | Sparse `csr_matrix`, no densification | **< 500 ms** on a laptop CPU |
| Preprocessing (50 resumes) | Single model load + `nlp.pipe(batch_size=32)` | ~1–3 s (model-bound, one-time load cached) |
| Memory | Streaming 64 KiB chunks, immediate `del` of bytes, O(nnz) TF-IDF | Flat vs. doc count; 10 MB/file cap |
| Time complexity | **O(N·M)**, N = vocab, M = doc length | Linear scaling in batch size |
| Space complexity | **O(nnz)** sparse | No `toarray()`/`todense()` on TF-IDF |

Tips: lower *TF-IDF max features* (sidebar) for even faster runs;
keep Parser/NER disabled in spaCy (already done) for ~2× speedup.

---

## 🔒 Security Considerations

| Threat | Mitigation |
|---|---|
| Path traversal (`../../etc/passwd`) | `sanitize_filename()` — basename + `[^A-Za-z0-9_.-]` strip |
| Arbitrary file execution / macros | Text-only extraction (pdfplumber / PyPDF2 / python-docx); code never executed |
| Malicious file types | Extension whitelist (`.pdf`, `.docx`) **+ magic-byte signature check** (`%PDF`, `PK..`); spoofed types rejected |
| Injection via extracted text | `sanitize_text()` strips control chars / null bytes, caps length |
| Memory-exhaustion DoS | 10 MB/file cap, 50-file batch cap, chunked reads |
| PII leakage / persistence | **Zero disk writes** — `io.BytesIO` only; buffers `del`-eted; optional email/phone/URL redaction |
| Keyword stuffing | Sublinear TF + L2 norm dampens gaming |

> ⚠️ This tool produces **decision support**, not decisions — always keep a
> human in the loop for hiring. TF-IDF measures lexical overlap, not true
> competence; review top candidates manually.

---

## 🧪 Quick Smoke Test

```bash
python -c "import utils.parser, utils.preprocessor, utils.matcher; print('imports OK')"
python -m compileall -q app.py utils/ && echo "syntax OK"
pytest -q   # full unit suite: parser, preprocessor, matcher
```

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `blis/py.pyx` Cython errors (`cannot be cast to pointers`, `no suitable method found`) on `pip install spacy==3.7.5` | Old `spacy 3.7.5` → `blis 0.7.11` source build is incompatible with **Python 3.12+ / NumPy 2 / Cython 3** (tries to compile from sdist, no wheel) | **Use Python 3.10/3.11** (`/usr/bin/python3.11 -m venv .venv && pip install -r requirements.txt`), **or** use the modernized `requirements.txt` in this repo (`spacy>=3.7,<4`, `numpy>=1.26,<3`) which installs prebuilt wheels on 3.10–3.14 |
| `spaCy model 'en_core_web_sm' not found` | Model not installed | `python -m spacy download en_core_web_sm` |
| `PDF is password-protected / encrypted` (`st.error`) | Uploaded PDF needs a password | Re-export an unprotected copy and re-upload |
| `appears to contain scanned images` (`st.warning`, skipped) | Image-only PDF/DOCX, `< 10` chars of selectable text | Upload a text-based export or run OCR first |
| All scores `0.0%` + zero-overlap warning | JD and resumes share no tokens after preprocessing | Use a more detailed JD; check resumes have selectable text |
| `Unsupported file type` / signature rejection | Spoofed extension (e.g. `.exe` → `.pdf`) or bad type | Upload genuine `.pdf` / `.docx` only |
| `exceeds 10 MB limit` | Oversize file | Compress / split the file, or raise `MAX_FILE_SIZE_BYTES` in `utils/parser.py` |

---

## 📌 Tech Stack

Python 3.10+ · Streamlit · spaCy (`en_core_web_sm`) · scikit-learn
(`TfidfVectorizer`, `cosine_similarity`) · scipy sparse · pdfplumber /
PyPDF2 · python-docx · pandas · matplotlib · pytest

---

## 📄 License

MIT — free for personal and commercial use. Candidate data you upload
remains yours; the app never transmits or stores it.
