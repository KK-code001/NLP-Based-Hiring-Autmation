# Explainable Resume–JD Matching System
## Architecture & Complete Change Report

---

## 1. System Architecture

The system follows a **Hybrid NLP + Machine Learning pipeline**, matching the architecture originally proposed in the project's PS-1 document:

```
Resume + Job Description (as FILES: PDF / DOCX / JPG / PNG)
        |
        v
Multi-Format Text Extraction
   - PDF: text layer, auto-fallback to OCR if scanned
   - DOCX: paragraphs + table cells
   - JPG/PNG: OCR (Tesseract)
        |
        v
NLP Information Extraction
   - Skill extraction (dictionary + fuzzy matching)
   - Experience extraction (regex-based)
   - Education extraction
   - Domain identification
   - Semantic similarity (Sentence-BERT, TF-IDF fallback if offline)
        |
        v
Feature Generation
   - skill_match_score, matched/missing skill counts
   - experience_match_score, experience_ratio (uncapped)
   - domain_match, semantic_similarity_score
        |
        v
Candidate Matching Engine
   - Random Forest classifier (Match / Partial Match / No Match)
   - Priority Engine: deterministic safety-net rules that override
     model confidence for known high-risk patterns
        |
        v
Explainable AI Module (SHAP TreeExplainer)
   - Per-candidate reasoning: which factors drove the prediction
        |
        v
Recommendation Engine
   - Missing skills -> curated certification/course suggestions
        |
        v
Final Report
   - Prediction, confidence, human-review flag + reason,
     matched/missing skills, experience/domain analysis,
     SHAP reasoning, recommendations
```

**Design choice — Random Forest over Logistic Regression:** both were
tested with stratified 5-fold cross-validation and performed comparably
(~70% accuracy). Random Forest was selected because it pairs with SHAP's
`TreeExplainer` (exact, not approximate), which is central to the
project's explainability goal — not because it was meaningfully more
accurate.

**Known, documented limitation:** the dataset's `match_score`/`match_label`
are synthetic (a deterministic rule, not human-annotated), and the
extracted features explain only ~47% of `match_score`'s variance. The
system is honestly framed as reconstructing a proxy label, not "true"
recruiter judgment.

---

## 2. Complete Change Log

### Phase 1 — Initial pipeline build
| # | Change | Why |
|---|---|---|
| 1 | Consolidated the two original notebooks (cleaning + feature engineering) into one reusable `pipeline.py` module | Original notebooks only worked as a batch process with manual CSV re-uploads between steps; couldn't handle one new resume on demand |
| 2 | Added PDF text extraction (`pdfplumber`) | Original pipeline had no file-input capability at all |

### Phase 2 — Bug fixes found by testing against the real dataset
| # | Change | Why |
|---|---|---|
| 3 | Fixed symbol-stripping bug in `clean_text()` | Cleaning silently stripped `+`, `#`, `.`, `&` before skill matching, breaking detection of "C++", "C#", "node.js", "FP&A" even though they were in the dictionary |
| 4 | Expanded skill dictionary ~160 → ~280 terms | Scanned all 500 JD "Required Skills" sections; found 28 JDs with **zero** detectable skills. Added missing cybersecurity, UX/design, sales, HR, and finance terms |
| 5 | Generalized `&`/`-` handling | Same root cause as #3, applied to compound terms like "diversity & inclusion", "cross-selling" |
| 6 | Added fuzzy matching for typos/variants | Catches "communications"→"communication", "labour laws"→"labor law". Tuned twice: first attempt (88% threshold) produced false positives ("trust"→"rust", "market"→"marketo"); fixed with a 90% threshold + same-first-letter filter + short-skill blocklist |
| 7 | Added PDF error handling (`ScannedPDFError`, `CorruptPDFError`) | Scanned/corrupt PDFs previously crashed with a raw stack trace |

**Measured impact of #3–#6:** resumes/JDs with zero detected skills went from 9/500 to 0/500; `skill_match_score`'s correlation with `match_score` rose from 0.581 to 0.724 (became the single strongest feature).

### Phase 3 — Modeling
| # | Change | Why |
|---|---|---|
| 8 | Added `resume_experience_mentioned`/`jd_experience_mentioned` flags | So "unknown experience" is a visible signal, not silently erased |
| 9 | Trained Random Forest vs Logistic Regression, stratified 5-fold CV | Compared honestly rather than assuming a model choice |
| 10 | Added confidence-based human review (threshold = 0.55) | The actual fix for the "Partial Match" class being hardest to predict — routes uncertain cases to a human instead of forcing a guess. Threshold chosen at the point where accuracy gains (69.6%→73.0%) start to plateau relative to review workload (16.4% of candidates) |
| 11 | Added SHAP `TreeExplainer` | Per-candidate explainability — the project's core differentiator |
| 12 | Built the Recommendation Engine (~45 curated skill→certification mappings) | Turns "missing skills" into actionable guidance, not just a rejection |

### Phase 4 — Priority Engine (verified from an externally-provided changelog, one bug found and fixed)
| # | Change | Why |
|---|---|---|
| 13 | Sentinel imputation (`-1.0` instead of the training median) for missing experience scores | The median happened to be `1.0`, meaning "we don't know this candidate's experience" was being scored identically to "perfect experience match" — a real, dangerous bug |
| 14 | Added uncapped `experience_ratio` + `is_overqualified` flag | `experience_match_score` is capped at 1.0, so a 16-year candidate for a 1-year role looked identical to an exact match. The report now surfaces an explicit overqualification alert |
| 15 | Added `has_sufficient_evidence()` gate | A confidence score alone doesn't reflect how much real evidence backed a prediction; near-empty resumes could score confidently by accident. Now requires both sufficient matched+missing skill count AND a minimum number of skills detected on the resume itself |
| 16 | Fixed domain "Unknown == Unknown" bug | Two resumes/JDs that couldn't be classified into any domain were being counted as a domain **match** |
| 17 | Added "Unstated Experience vs. Senior Role" deterministic safety net | A fresher resume could get silently matched to a Principal-level role at 65%+ confidence. This rule **always** forces human review for this pattern, regardless of model confidence — mirrors how real HR/ATS systems combine statistical models with explicit business rules for rare, high-stakes edge cases |
| 18 | **Found and fixed:** `FEATURE_LABELS` was missing an entry for the new senior-role feature | Verified independently by testing a fresher-vs-Principal-role case directly — confirmed this would crash `build_final_report()` with a `KeyError` whenever SHAP ranked that feature in its top 4 |

### Phase 5 — Fixes from real dry-run testing (Vivaan Mishra resume vs. UnitedHealth JD)
| # | Change | Why |
|---|---|---|
| 19 | Added LLM/RAG/Agentic AI vocabulary to the skill dictionary | These terms were completely invisible to the system; a real JD saturated with them showed zero skill detection in that section |
| 20 | Added "implied skills" logic (e.g. GitHub → Git, React → JavaScript) | A resume listing "GitHub" was wrongly marked as **missing** "Git," even though having a GitHub profile demonstrates git usage |
| 21 | Added TF-IDF cosine-similarity fallback for semantic similarity | Sentence-BERT requires downloading a model from huggingface.co; in network-restricted environments this crashed the pipeline. Now falls back automatically, with a one-time warning and an explicit `semantic_similarity_method` field so the report is never silently misleading about which method was used |

### Phase 6 — Multi-format file input
| # | Change | Why |
|---|---|---|
| 22 | Added DOCX text extraction (paragraphs + table cells) | Resumes/JDs arrive as Word documents too, not just PDFs |
| 23 | Added OCR support (Tesseract) for scanned PDFs and JPG/PNG images | Previously, scanned/photographed resumes were rejected outright |
| 24 | Built a unified `extract_text_from_file()` that auto-detects format from the extension | One function handles PDF/DOCX/JPG/PNG for both resume and JD, in any combination |
| 25 | **Caught and fixed a self-introduced regression:** the multi-format merge briefly reverted the entire Priority Engine (Phase 4) | Found by re-running the full pipeline end-to-end after the merge and hitting a `KeyError` for missing feature columns — traced to merging the new file-support code onto the wrong base version. Re-merged onto the correct base and re-verified |

### Phase 7 — Usability fixes
| # | Change | Why |
|---|---|---|
| 26 | Added `extract_candidate_name()` — auto-detects the candidate's name from the top of the resume | Removed the need to manually type the candidate's name every run; verified against a real resume PDF |
| 27 | Pinned `Pillow==10.4.0` in the setup cell | Installing OCR libraries (`pytesseract`, `pdf2image`) pulled in an incompatible Pillow version that broke SHAP/matplotlib (`ImportError: cannot import name '_Ink'`) |
| 28 | Produced a clean, code-only Colab notebook with all built-in demo/test cells removed | So the first real input the pipeline processes is the user's own upload, not a pre-filled example |

---

## 3. Verification Methodology

Every change in this report was **actually executed and tested**, not just written:
- All notebooks were run end-to-end after each change (with a local stand-in only for the one function — Sentence-BERT — that needs external network access unavailable in the testing sandbox)
- Bug claims (e.g. the symbol-stripping bug, the `FEATURE_LABELS` crash, the Priority Engine regression) were reproduced with a failing test *before* being fixed, and re-verified passing *after*
- Dictionary/fuzzy-matching changes were measured against the real 500-row dataset (before/after coverage and correlation numbers), not assumed
