# Explainable AI-Based Resume and Job Description Matching System

## Overview

This project presents an Explainable AI-Based Resume and Job Description Matching System designed to improve transparency, fairness, and trust in automated candidate screening.

Traditional Applicant Tracking Systems (ATS) often provide only acceptance or rejection decisions without explaining the reasoning behind those decisions. The proposed system addresses this limitation by combining Natural Language Processing (NLP), Semantic Similarity Analysis, Feature Engineering, and Explainable AI techniques to evaluate candidate-job compatibility and provide meaningful explanations.

The system analyzes resumes and job descriptions, extracts relevant information, generates matching features, predicts candidate suitability, and produces interpretable recommendations.

---

## Dataset

The primary dataset consists of approximately 500 Resume–Job Description pairs.

### Dataset Features

* Resume Text
* Job Description
* Match Score
* Match Label

### Match Labels

* Match
* Partial Match
* No Match

Example:

**Input**

Resume Text

Job Description

**Output**

Match Score = 0.55

Match Label = Partial Match

---

## System Architecture

```text
Resume Text + Job Description
│
▼
Data Preprocessing
│
▼
NLP Information Extraction
│
├── Skills Extraction
├── Experience Extraction
├── Education Extraction
└── Domain Identification
│
▼
Feature Generation
│
├── Skill Match Score
├── Experience Match Score
├── Semantic Similarity Score
└── Domain Match
│
▼
Candidate Matching Engine
│
▼
Suitability Prediction
│
├── Match
├── Partial Match
└── No Match
│
▼
Explainable AI Module
│
├── Matched Skills
├── Missing Skills
├── Key Matching Factors
└── Prediction Explanation
│
▼
Recommendation Engine
│
├── Suggested Skills
├── Suggested Certifications
├── Experience Improvements
└── Learning Recommendations
│
▼
Final Report
```

---

## Data Preprocessing

The preprocessing pipeline prepares raw resume and job description text for NLP analysis.

### Operations Performed

* Duplicate removal
* Email removal
* Phone number removal
* URL removal
* Text normalization
* Lowercase conversion
* Special character removal

### Example

Before:

```text
Name: John Doe
Email: john@gmail.com
Phone: +91-9876543210
```

After:

```text
john doe
```

---

## NLP Information Extraction

The system extracts structured information from unstructured resume and job description text.

### Skills Extraction

Examples:

* Python
* SQL
* SAP
* CFA
* Financial Reporting
* Machine Learning
* TensorFlow

Generated Features:

* resume_skills
* jd_skills

### Experience Extraction

Examples:

* 3 years
* 5 years
* 8 years

Generated Features:

* resume_experience
* jd_experience

### Education Extraction

Examples:

* B.Tech
* MBA
* CA
* CFA
* M.S
* PhD

Generated Features:

* resume_education
* resume_education_level

Education Levels:

* Bachelor
* Master
* Professional
* Diploma
* Doctorate

### Domain Identification

Examples:

* Finance
* HR
* Data Science
* Marketing
* DevOps
* Cyber Security
* Product Management
* Sales
* Software Development

Generated Features:

* resume_domain
* jd_domain
* domain_match

---

## Feature Engineering

The extracted information is transformed into quantitative features used for candidate evaluation.

### Skill Match Score

Measures overlap between resume skills and job requirements.

Example:

Required Skills = 10

Matched Skills = 7

Skill Match Score = 70%

---

### Experience Match Score

Measures compatibility between candidate experience and required experience.

Example:

Candidate Experience = 5 Years

Required Experience = 8 Years

Experience Match Score = 62.5%

---

### Semantic Similarity Score

Measures contextual similarity between resumes and job descriptions.

Model Used:

Sentence-BERT (all-MiniLM-L6-v2)

Output Range:

* 0.00 → Completely Different
* 1.00 → Highly Similar

Example:

Similarity Score = 0.82

Interpretation:

Highly Relevant Candidate

---

## Candidate Matching Engine

The Candidate Matching Engine combines generated features to evaluate candidate suitability.

### Input Features

* Skill Match Score
* Experience Match Score
* Semantic Similarity Score
* Domain Match

### Output Classes

* Match
* Partial Match
* No Match

Potential Machine Learning Models:

* Random Forest
* XGBoost
* Logistic Regression

---

## Explainable AI Module

The Explainable AI component provides transparency into model decisions.

Instead of displaying:

```text
Candidate Rejected
```

The system explains:

### Example

Match Score: 62%

Matched Skills:

✓ CFA

✓ Financial Planning

✓ Valuation

Missing Skills:

✗ SAP

✗ QuickBooks

✗ GAAP

Experience Gap:

5 Years vs Required 8 Years

Decision:

Partial Match

Reason:

Strong finance background but lacks ERP and reporting tools.

Techniques:

* SHAP
* Feature Importance Analysis

---

## Recommendation Engine

The Recommendation Engine provides actionable feedback for candidate improvement.

### Example Recommendations

Suggested Skills:

* SAP
* Financial Reporting
* Treasury Management

Suggested Certifications:

* SAP Certification
* CFA Level II

Suggested Learning Areas:

* Financial Modeling
* Regulatory Compliance

The objective is to transform the system from a rejection engine into a guidance system.

---

## Final Output Report

The final report generated by the system contains:

* Candidate Name
* Overall Match Score
* Match Classification
* Matched Skills
* Missing Skills
* Experience Analysis
* Education Analysis
* Semantic Similarity Score
* Prediction Explanation
* Improvement Recommendations

---

## Technologies Used

* Python
* Pandas
* NumPy
* Regular Expressions (Regex)
* NLTK
* spaCy
* Sentence Transformers
* Scikit-Learn
* SHAP
* Google Colab
* Git & GitHub

---

## Research Contribution

This project introduces transparency into AI-based candidate screening by combining:

* NLP-based Information Extraction
* Semantic Resume–Job Matching
* Explainable AI Techniques
* Personalized Candidate Recommendations

The system not only predicts candidate suitability but also explains the reasoning behind its decisions, improving trust, fairness, interpretability, and usability in automated recruitment systems.
