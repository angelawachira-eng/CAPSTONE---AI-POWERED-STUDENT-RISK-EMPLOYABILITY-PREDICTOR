# CAPSTONE---AI-POWERED-STUDENT-RISK-EMPLOYABILITY-PREDICTOR
This project develops an AI-powered system using Machine Learning and NLP to predict students’ academic risk and employability readiness. It analyzes academic records, skills, and resumes to identify skill gaps, assess job-market preparedness, and provide personalized recommendations for improvement.


## Overview

Higher education plays a critical role in socio-economic mobility and personal development. This project builds a multi-layered AI system to help institutions identify at-risk students early and assess graduate employability. It covers four prediction tasks across three notebooks:

| Task | Type | Target |
|---|---|---|
| Dropout Prediction | Multiclass classification | Graduate / Enrolled / Dropout |
| Academic Risk | Multiclass classification | High Performance / Average / At Risk |
| Employability | Multiclass classification | High / Medium / Low Readiness |
| Career Path (NLP) | Multiclass classification | Job category from resume text |

---

## Business and Data Understanding

### Stakeholder Audience

| Stakeholder | Interest |
|---|---|
| University Administration | Improve institutional retention rates and reputation |
| Academic Advisors | Receive early-warning signals to support struggling students |
| Students | Self-assess academic standing and career readiness |
| Career Services | Align student skills with industry demands |
| Employers | Identify job-ready graduate candidates |

### Dataset Choice

Four complementary datasets were combined to address both the academic risk and employability dimensions of the problem:

| Dataset | Records | Key Features |
|---|---|---|
| Dropout (`data.csv`) | 4,424 | Demographics, enrolment details, semester grades, financial status |
| Academic Risk | Multi-institution | GPA, absences, hours studied, engagement score |
| Employability | 1,200 | GPA, internships, certifications, soft skills, research participation |
| Resume (`Resume.csv.gz`) | ~2,400 | Resume text, job category, experience |

The dropout dataset provides a labelled, real-world student outcome — making it the backbone of the risk prediction task. The employability and resume datasets extend the project into career readiness, ensuring the system is useful beyond just identifying who drops out.

---

## Modeling

Three algorithms were applied across the four tasks:

| Domain | Algorithms | Features Used |
|---|---|---|
| Dropout Prediction | Random Forest, XGBoost | Age, admission grade, units approved/enrolled, scholarship status, financial flags |
| Academic Risk | Random Forest, XGBoost | GPA, absences, hours studied, engagement score, study efficiency |
| Employability | Random Forest, XGBoost | GPA, tech score, soft score, career score averages |
| Resume NLP | Logistic Regression, Random Forest | 10 TF-IDF LSA topic components + `has_experience` flag |

Class imbalance in the dropout and academic datasets was addressed with **SMOTE** before training. All features were scaled and encoded in the EDA notebook. A global random seed of `42` is used throughout for reproducibility.

---

## Evaluation

Models were evaluated using **weighted F1-score** to account for class imbalance. Results were validated with **5-fold stratified cross-validation** to confirm consistency across data splits, and **RandomizedSearchCV** (20 iterations, 5-fold) was used for hyperparameter tuning. **SHAP values** were used to interpret feature importance and explain model decisions.

Key findings from evaluation:

- **1st-semester grades and units approved** are the strongest early dropout signals — dropouts average 3.4 grade points lower than graduates from their very first semester.
- **Financial factors** (tuition arrears, debt status) are highly predictive — students up-to-date on fees graduate at ~82% vs ~35% for dropouts.
- **Scholarship holders** graduate at nearly double the rate of non-holders.
- **GPA alone is insufficient** for employability prediction — technical skills, soft skills, and research participation contribute more to the competitiveness score.

---

## Conclusion

All four project objectives were successfully met:

1. **Dropout classification** — Random Forest and XGBoost accurately distinguish Dropouts from Graduates, confirming that early academic momentum and financial stability are the strongest retention anchors.

2. **Employability assessment** — Classifiers accurately categorise students into Low, Medium, and High readiness tiers, showing that technical and soft skills outweigh GPA in determining job market readiness.

3. **Key risk drivers identified** — SHAP analysis and EDA correlation heatmaps revealed that Age at Enrollment, Debtor Status, 1st Semester Units Approved, GPA, and Hours Studied are the highest quantifiable drivers of student risk.

4. **Deployable models** — The best-performing model per task is saved as a `.pkl` file, ready for integration into a real-time prediction interface (e.g., a Streamlit app) for use by advisors and career services.
