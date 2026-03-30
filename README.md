# NHANES Prediabetes Prediction API

**Authors:** Zhalae Daneshvari, Jolene Ie, Nick Johnson
**Course:** INFO 4940 — Fall 2025

---

## Overview

This project develops and deploys a **machine learning API** that predicts prediabetes likelihood using self-reported survey data from the CDC's National Health and Nutrition Examination Survey (NHANES, Aug 2021–Aug 2023).

Prediabetes is widespread yet often undiagnosed — many adults never undergo routine blood testing. A survey-based model that flags individuals as *likely prediabetic* provides a low-cost, scalable first step: it allows clinics and community health programs to focus lab resources on those most at risk, promote earlier diagnosis, and help prevent progression to Type 2 diabetes.

The final product is a **logistic regression classifier** wrapped in a Vetiver FastAPI service, containerized with Docker, and hosted on the Cornell Applied ML Server.

**Live API:** [`http://appliedml.infosci.cornell.edu:2222`](http://appliedml.infosci.cornell.edu:2222)
**Interactive Docs (Swagger UI):** [`http://appliedml.infosci.cornell.edu:2222/docs`](http://appliedml.infosci.cornell.edu:2222/docs)

---

## Repository Structure

```
proj-01-dank-corgi/
│
├── data-raw/                        # NHANES datasets and cleaned features
│   ├── nhanes2123_surveyonly_clean.csv
│   └── other supporting CSVs
│
├── plots/                           # ROC, PR, calibration, and feature importance plots
│
├── app.py                           # FastAPI entry point (Vetiver API)
├── Dockerfile                       # Docker build file for API deployment
├── vetiver_requirements.txt         # API dependencies
│
├── api.qmd                          # Model training, deployment & testing script
├── report.qmd                       # Final written report
├── report-model-code.qmd            # Core model training code
├── model-card.qmd                   # Vetiver model card for transparency
├── proposal.qmd                     # Original project proposal
│
├── plots/                           # ROC curve, PR curve, calibration, feature importance
├── _team-agreement.md               # Collaboration agreement
└── README.md                        # You are here
```

---

## Data

**Source:** [Continuous NHANES (CDC/NCHS)](https://wwwn.cdc.gov/nchs/nhanes/), Aug 2021–Aug 2023 — public domain, fully de-identified, approved under NCHS IRB protocols.

- **5,099 adult respondents** (age ≥ 18), one row per person identified by `SEQN`
- **Target:** `prediabetes_flag` = 1 if fasting plasma glucose (`LBXGLU`) ∈ [100–125 mg/dL] or glycohemoglobin (`LBXGH`) ∈ [5.7–6.4%]; all lab variables removed from predictors after target creation to prevent leakage
- **Class prevalence:** ~42% prediabetic

**Predictors (survey-only):**

| Type | Features |
|------|----------|
| Demographic | Age, sex, race/ethnicity |
| Anthropometric | Self-reported BMI, one-year weight change |
| Behavioral | Sleep hours, sedentary minutes, alcohol use, smoking history |
| Medical history | Hypertension, high cholesterol, cardiovascular disease, asthma, thyroid, liver |
| Mental health | PHQ-9 items (poor appetite, feeling down) |

### Data Collection

NHANES uses a **stratified multistage probability design** to yield a nationally representative sample of U.S. civilians. Data collection involves interviews in participants' homes, physical exams in mobile exam centers, and standardized lab measurements. Questionnaire modules are self-reported; lab measures (glucose, HbA1c) are obtained via blood draws.

### Preprocessing

1. Merged `.XPT` files from DEMO, WHQ, SLQ, SMQ, MCQ, and DPQ modules via `SEQN`
2. Filtered to adults (≥18) with valid glucose or HbA1c; excluded diagnosed diabetics and medication users
3. Dropped redundant `_code` columns, identifiers, lab predictors, and high-missingness items (>30–40%)
4. Set implausible numeric values (e.g., sedentary_minutes > 2000/day) to `NaN`
5. **Numeric:** median imputation + missingness indicator flags, StandardScaler
6. **Categorical:** `"Unknown"` imputation, one-hot encoding with `min_frequency=0.02` to collapse rare levels
7. Derived features: `bmi_from_selfreport`, `weight_change_1y_lb`, `ever_smoker`, `current_smoker`

---

## Model

**Type:** Logistic Regression with L2 regularization, cross-validated (`LogisticRegressionCV`)
**Framework:** scikit-learn `Pipeline` → Vetiver → FastAPI → Docker

Four model families were evaluated: L2 logistic regression, elastic net, random forest, and histogram gradient boosting. All were evaluated using **stratified 5-fold cross-validation** on the training set, scored by ROC AUC, PR AUC, and Brier score.

### Model Comparison (5-fold CV, training set)

| Model | Accuracy | Bal. Acc | ROC AUC | PR AUC | Brier |
|---|---|---|---|---|---|
| **LogReg L2 — selected** | **0.656** | **0.640** | **0.717** | **0.615** | **0.210** |
| LogReg Elastic Net + Calib | 0.656 | 0.638 | 0.707 | 0.605 | 0.214 |
| LogReg Elastic Net | 0.658 | 0.641 | 0.707 | 0.604 | 0.214 |
| Random Forest (tuned) | 0.651 | 0.654 | 0.699 | 0.590 | 0.218 |
| HistGradientBoosting T1 | 0.644 | 0.631 | 0.690 | 0.569 | 0.219 |
| HistGradientBoosting T2 | 0.629 | 0.611 | 0.676 | 0.561 | 0.223 |
| Null (majority class) | 0.577 | — | 0.500 | 0.423 | 0.423 |

The L2 logistic regression was selected for its best overall balance of discrimination and calibration, and for interpretability — each coefficient maps directly to a survey feature's contribution to predicted risk.

### Resampling Strategy

- **Stage 1:** 80/20 stratified train/test split (`random_state=42`)
- **Stage 2:** Stratified 5-fold CV within the training set for all tuning
- **Stage 3:** Decision threshold calibrated on a 20% validation split of training data to achieve **recall ≥ 0.80**, prioritizing sensitivity in a screening context (false negatives are more harmful than false positives)

---

## Results

### Test Set Performance

- **ROC AUC:** 0.717 — **PR AUC:** 0.615 — **Brier score:** 0.210
- **Decision threshold:** 0.3833
  - Precision: **0.552** | Recall: **0.763** | F1: **0.641**
  - Confusion matrix: TN 322 · FP 267 · FN 102 · TP 329

The model generalizes well to held-out data. At the chosen threshold, 329 truly at-risk individuals are correctly flagged (true positives), and only 102 are missed (false negatives). The 267 false positives represent individuals flagged for follow-up testing who may not have prediabetes — an acceptable trade-off in a screening context.

### Group Performance (test set, thresholded)

| Group | n | Prevalence | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Female | 552 | 0.379 | 0.537 | 0.732 | 0.619 |
| Male | 468 | 0.474 | 0.566 | 0.793 | 0.660 |
| Non-Hispanic White | 627 | 0.396 | 0.518 | 0.754 | 0.614 |
| Non-Hispanic Black | 99 | 0.535 | 0.636 | 0.792 | 0.706 |
| Mexican American | 82 | 0.500 | 0.667 | 0.780 | 0.719 |
| Non-Hispanic Asian | 62 | 0.435 | 0.525 | 0.778 | 0.627 |
| Other Hispanic | 99 | 0.394 | 0.526 | 0.769 | 0.625 |
| Other/Multiracial | 51 | 0.451 | 0.708 | 0.739 | 0.723 |

Recall is consistently high across groups. Precision varies somewhat by sex and race/ethnicity, largely tracking with differences in group-level prevalence.

### Top Predictors

Age, self-reported BMI, and hypertension history are the strongest predictors — reflecting established links between aging, cardiometabolic risk, and glucose dysregulation. Race/ethnicity contributes meaningfully, primarily reflecting structural health disparities and known population-level risk differences. Behavioral and mental health features (sedentary time, sleep, weight change, PHQ-9 items) contribute smaller but complementary signal.

Partial dependence plots confirm intuitive monotone relationships: predicted prediabetes probability increases steadily with age and BMI. Sedentary minutes and drinks per day show slight negative slopes, likely reflecting confounding — younger adults report more sedentary time and alcohol use but have lower BMI and lower baseline prevalence.

---

## API Workflow

### 1. Train and Pin the Model

```bash
quarto render api.qmd
```

This script:
1. Loads and cleans the NHANES dataset
2. Trains the logistic regression pipeline
3. Creates a `VetiverModel` and pins it to GCS (`info-4940-models/ji92/`)
4. Auto-generates Docker files for deployment

### 2. Build and Run the Docker Container

```bash
docker build -t prediabetes-model-3 .
docker run -p 2222:2222 prediabetes-model-3
```

You should see:
```
Uvicorn running on http://0.0.0.0:2222
```

Then navigate to `http://appliedml.infosci.cornell.edu:2222/docs` for the interactive Swagger UI.

### 3. Test the API

```python
import pandas as pd
import requests
import json

API_URL = "http://appliedml.infosci.cornell.edu:2222/predict"

df = pd.read_csv("data-raw/nhanes2123_surveyonly_clean.csv")

if "prediabetes_flag" in df.columns:
    df = df.drop(columns=["prediabetes_flag"])

for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna("Unknown").astype(str)

sample_rows = df.sample(5, random_state=42).to_dict(orient="records")

response = requests.post(
    API_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(sample_rows)
)

print(response.json())
# Example: {"predict": [0, 0, 1, 0, 0]}
# 0 → Not Prediabetic  |  1 → Prediabetic
```

### Prediction Schema

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

| Response | Meaning |
|----------|---------|
| `0` | Model predicts not prediabetic |
| `1` | Model predicts likely prediabetic |

---

## How to Reproduce

### Option A — Run the Full Pipeline

```bash
git clone https://github.com/NickMJohnson/proj-01-dank-corgi.git
cd proj-01-dank-corgi
pip install -r vetiver_requirements.txt
quarto render api.qmd
docker build -t prediabetes-model-3 .
docker run -p 2222:2222 prediabetes-model-3
```

### Option B — Use the Live API (no build needed)

Send JSON to `POST http://appliedml.infosci.cornell.edu:2222/predict`

See `/docs` for the full schema and live testing interface.

---

## Intended Use & Limitations

**Intended use:** Survey-based pre-screening to flag adults who may benefit from confirmatory glucose or HbA1c testing. This is a **decision-support tool**, not a standalone diagnostic. Suitable for research, education, and public-health planning.

**Limitations:**
- Predictions are **associational, not causal** — the model describes population patterns, not individual risk pathways
- **Self-report bias** affects key features like sleep, sedentary time, and alcohol use; respondents may understate unhealthy behaviors
- **Precision varies by subgroup** (sex, race/ethnicity), consistent with prevalence differences — monitor to prevent systematic disparities
- Trained on 2021–2023 data; performance may drift as health behaviors, population composition, or measurement protocols change
- **External validity** is limited to undiagnosed adults — not appropriate for pediatric populations or already-diagnosed diabetics
- Structured missingness means individuals with more missing data (often higher-risk groups) are underrepresented, which could make predictions overly optimistic for marginalized populations

**Future improvements:**
- Train across multiple NHANES cycles for better temporal generalizability
- Periodic recalibration (Platt or isotonic scaling) as new data become available
- Group-specific thresholds if precision disparities persist
- External validation on community health surveys outside NHANES

---

## Authentication & Storage

Models are pinned to a shared GCS bucket: `info-4940-models/ji92/`

Authentication uses a Google Cloud service account key file. The Docker container sets this via:

```bash
ENV GOOGLE_APPLICATION_CREDENTIALS="/vetiver/app/service-auth.json"
```

> **Note:** `service-auth.json` is excluded from version control via `.gitignore`. You must supply your own credentials to re-pin the model.

---

## Citation

Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS). *National Health and Nutrition Examination Survey Data, 2021–2023.* Hyattsville, MD: U.S. Department of Health and Human Services.
Available at: https://wwwn.cdc.gov/nchs/nhanes/
