# NHANES Prediabetes Prediction API

**Authors:** Zhalae Daneshvari, Jolene Ie, Nick Johnson
**Course:** INFO 4940 — Fall 2025

---

## Overview

This project develops and deploys a **machine learning API** that predicts prediabetes likelihood using self-reported survey data from the CDC's National Health and Nutrition Examination Survey (NHANES, Aug 2021–Aug 2023).

Prediabetes is widespread yet often undiagnosed — many adults never undergo routine blood testing. A survey-based model that flags individuals as *likely prediabetic* provides a low-cost, scalable first step: it allows clinics and community health programs to focus lab resources on those most at risk, promote earlier diagnosis, and help prevent progression to Type 2 diabetes.

The final product is a **logistic regression classifier** wrapped in a Vetiver FastAPI service, containerized with Docker, and hosted on the Cornell Applied ML Server.

---

## Data

**Source:** [Continuous NHANES (CDC/NCHS)](https://wwwn.cdc.gov/nchs/nhanes/), Aug 2021–Aug 2023 — public domain, fully de-identified.

- **5,099 adult respondents** (age ≥ 18), one row per person identified by `SEQN`
- **Target:** `prediabetes_flag` = 1 if fasting glucose (100–125 mg/dL) or HbA1c (5.7–6.4%); all lab variables removed from predictors after target creation
- **Prevalence:** ~42% prediabetic in the sample

**Predictors (survey-only):**

| Type | Features |
|------|----------|
| Demographic | Age, sex, race/ethnicity |
| Anthropometric | Self-reported BMI, one-year weight change |
| Behavioral | Sleep hours, sedentary minutes, alcohol use, smoking history |
| Medical history | Hypertension, high cholesterol, cardiovascular disease, asthma, thyroid, liver |
| Mental health | PHQ-9 items (poor appetite, feeling down) |

---

## Model

**Type:** Logistic Regression with L2 regularization, cross-validated (`LogisticRegressionCV`)
**Framework:** scikit-learn pipeline with median imputation, standard scaling, and one-hot encoding

### Preprocessing

- **Numeric:** median imputation + missingness indicators, StandardScaler
- **Categorical:** `"Unknown"` imputation, one-hot encoding with `min_frequency=0.02` to collapse rare levels

### Model Comparison (5-fold CV on training set)

| Model | Accuracy | ROC AUC | PR AUC | Brier |
|---|---|---|---|---|
| **LogReg L2 (selected)** | **0.656** | **0.717** | **0.615** | **0.210** |
| LogReg Elastic Net | 0.658 | 0.707 | 0.604 | 0.214 |
| Random Forest | 0.651 | 0.699 | 0.590 | 0.218 |
| HistGradientBoosting | 0.644 | 0.690 | 0.569 | 0.219 |
| Null (majority class) | 0.577 | 0.500 | 0.423 | 0.423 |

The L2 logistic regression was selected for its best combination of discrimination and calibration, and its interpretability — each coefficient maps directly to a survey feature's contribution to risk.

### Test Set Results

- **ROC AUC:** 0.717 — **PR AUC:** 0.615 — **Brier score:** 0.210
- **Decision threshold:** 0.3833 (tuned to achieve recall ≥ 0.80 for screening context)
  - Precision: 0.552 | Recall: 0.763 | F1: 0.641
  - Confusion matrix: TN 322, FP 267, FN 102, TP 329

### Top Predictors

Age, self-reported BMI, and hypertension history are the strongest predictors, reflecting the well-established links between aging, cardiometabolic risk, and glucose dysregulation. Behavioral features (sedentary time, sleep, weight change) contribute smaller but complementary signal.

---

## Intended Use & Limitations

**Intended use:** Survey-based pre-screening to flag individuals who may benefit from confirmatory glucose or HbA1c testing. This is a decision-support tool, not a standalone diagnostic.

**Limitations:**
- Predictions are associational, not causal
- Self-report bias affects features like sleep, alcohol use, and sedentary time
- Precision varies by subgroup (sex, race/ethnicity), consistent with prevalence differences
- Model is trained on 2021–2023 data; performance may drift as health behaviors or population composition change
- External validity is limited to undiagnosed adults — not applicable to pediatric or already-diagnosed populations

---

## Repository Structure

```
proj-01-dank-corgi/
├── data-raw/                        # NHANES datasets and cleaned features
│   ├── nhanes2123_surveyonly_clean.csv
│   └── other supporting CSVs
├── plots/                           # ROC, PR, calibration, and feature importance plots
├── app.py                           # FastAPI entry point (Vetiver API)
├── Dockerfile                       # Docker build file for API deployment
├── vetiver_requirements.txt         # API dependencies
├── api.qmd                          # Model training, deployment & testing script
├── report.qmd                       # Final written report
├── report-model-code.qmd            # Core model training code
├── model-card.qmd                   # Vetiver model card
└── proposal.qmd                     # Original project proposal
```

---

## API

**Endpoint:** `http://appliedml.infosci.cornell.edu:2222`
**Interactive docs (Swagger UI):** `http://appliedml.infosci.cornell.edu:2222/docs`

### Reproducing the Pipeline

**1. Train and pin the model**

```bash
quarto render api.qmd
```

This loads NHANES data, trains the logistic regression pipeline, pins the model to GCS via Vetiver, and auto-generates Docker files.

**2. Build and run the container**

```bash
docker build -t prediabetes-model-3 .
docker run -p 2222:2222 prediabetes-model-3
```

**3. Send a prediction**

```bash
POST http://appliedml.infosci.cornell.edu:2222/predict
Content-Type: application/json
```

Example input:

```json
{
  "age_years": 43,
  "sex": "Female",
  "race_eth": "Non-Hispanic White",
  "bmi_from_selfreport": 26.3,
  "sleep_hours_weekdays": 7,
  "sedentary_minutes": 360,
  "drinks_per_day_12mo": 1,
  "ever_high_bp_ans": "Yes",
  "ever_high_chol_ans": "No",
  "smoked_100_cigs_ans": "No",
  "smoke_now_ans": "Not at all",
  "tried_lose_weight_past_year_ans": "Yes",
  "dpq_poor_appetite_ans": "No",
  "dpq_feeling_down_ans": "No",
  "ever_asthma_ans": "No",
  "ever_chf_ans": "No",
  "ever_heart_attack_ans": "No",
  "ever_stroke_ans": "No",
  "ever_thyroid_problem_ans": "No",
  "ever_liver_condition_ans": "No",
  "weight_change_1y_lb": -5,
  "ever_smoker": 0,
  "current_smoker": 0
}
```

**Response:**

| Value | Meaning |
|-------|---------|
| `0` | Model predicts not prediabetic |
| `1` | Model predicts likely prediabetic |

---

## Citation

Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS). *National Health and Nutrition Examination Survey Data, 2021–2023.* Hyattsville, MD: U.S. Department of Health and Human Services.
Data available at: https://wwwn.cdc.gov/nchs/nhanes/
