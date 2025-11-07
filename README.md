# NHANES Prediabetes Prediction API  
**Authors:** Zhalae Daneshvari, Jolene Ie, Nick Johnson  
**Date:** Fall 2025  


## Overview  
This project develops and deploys a **machine learning API** that predicts **prediabetes likelihood** using NHANES survey data.  
We trained a **logistic regression model with cross-validation**, deployed it as a **Vetiver FastAPI service** using **Docker**, and hosted it on the **Cornell Applied ML Server**.

The API is live at:
 **FastAPI Endpoint:**  
[`http://appliedml.infosci.cornell.edu:2222`](http://appliedml.infosci.cornell.edu:2222)

 **Interactive Docs (Swagger UI):**  
[`http://appliedml.infosci.cornell.edu:2222/docs`](http://appliedml.infosci.cornell.edu:2222/docs)

---

## Repository Structure  
proj-01-dank-corgi/

│

├── data-raw/ # NHANES datasets and cleaned features

│ ├── nhanes2123_surveyonly_clean.csv

│ └── other supporting CSVs

│

├── app.py # FastAPI entry point (Vetiver API)

├── Dockerfile # Docker build file for API deployment

├── vetiver_requirements.txt # API dependencies

│

├── api.qmd # Main API training & deployment & testing script

├── report.qmd # Final written report

├── report-model-code.qmd # Core model training code

├── model-card.qmd # Vetiver model card for transparency

├── proposal.qmd # Original project proposal

│

├── pr_curve_test.png, roc_curve_test.png

├── service-auth.json # GCS service credentials (private)

├── _team-agreement.md # Collaboration agreement

└── README.md # ← You are here


---

## Model Summary

- **Type:** Logistic Regression (L2 regularization, cross-validated)  
- **Framework:** scikit-learn  
- **Preprocessing:**  
  - Numeric: median imputation, standard scaling  
  - Categorical: constant imputation, one-hot encoding  
- **Target Variable:** `prediabetes_flag`  
- **Evaluation Metrics:**  
  - ROC-AUC ≈ 0.72  
  - Accuracy ≈ 0.66  

---

## API Workflow

### 1. Train and Pin the Model

The model is trained and pinned using **Vetiver**.  
Run this in Posit Workbench or locally:

```bash
quarto render api.qmd
```

This script:
1. Loads and cleans the NHANES dataset.
2. Trains a logistic regression model wrapped in a Pipeline.
3. Creates a Vetiver model (VetiverModel) and pins it to GCS (info-4940-models/ji92/).
4. Auto-generates Docker files for deployment.


### 2. Build and Run the Docker Container

In the ML server terminal:

```bash
docker build -t prediabetes-model-3 .
docker run -p 2222:2222 prediabetes-model-3
```

You should see a log line similar to 
Uvicorn running on http://0.0.0.0:2222


Then navigate to
http://appliedml.infosci.cornell.edu:2222/docs
to access the interactive API documentation.

### 3. Test the API Locally

Once the container is running, you can test predictions using the provided script inside the api.qmd file’s last section:

```python
import pandas as pd
import requests
import json

API_URL = "http://appliedml.infosci.cornell.edu:2222/predict"

# load and clean data
df = pd.read_csv("data-raw/nhanes2123_surveyonly_clean.csv")

if "prediabetes_flag" in df.columns:
    df = df.drop(columns=["prediabetes_flag"])

# clean schema to match prototype
for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna("Unknown").astype(str)

# pick random rows 
sample_rows = df.sample(5, random_state=42).to_dict(orient="records")

print(f"Sending {len(sample_rows)} rows to API...\n")

# POST to API 
response = requests.post(
    API_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(sample_rows)
)

# handle response
if response.status_code == 200:
    preds = response.json()
    print("Success. Raw API response:\n", preds, "\n")

    # detect response type automatically 
    if isinstance(preds, list) and all(isinstance(p, dict) for p in preds):
        # standard vetiver JSON format (list of dicts)
        for i, (inp, outp) in enumerate(zip(sample_rows, preds)):
            print(f"Row {i+1}:")
            print(f"  Input: age={inp['age_years']}, sex={inp['sex']}, BMI={inp['bmi_from_selfreport']}")
            print(f"  Predicted: {outp.get('predicted')} | Probability: {outp.get('pred_proba', 0):.3f}\n")

    elif isinstance(preds, list):
        # simple list of predictions
        for i, (inp, pred) in enumerate(zip(sample_rows, preds)):
            print(f"Row {i+1}:")
            print(f"  Input: age={inp['age_years']}, sex={inp['sex']}, BMI={inp['bmi_from_selfreport']}")
            print(f"  Predicted: {pred}\n")

    else:
        print("Unexpected format:", type(preds))

else:
    print(f"API Error {response.status_code}")
    print(response.text)
```

Example Output:
{"predict": [0, 0, 1, 0, 0]}


0 → Not Prediabetic
1 → Prediabetic

## Authentication & Storage

Models are pinned to a shared GCS bucket:
info-4940-models/ji92/

Authentication uses a service account key file:
service-auth.json

Docker containers include this key as an environment variable:
```bash
ENV GOOGLE_APPLICATION_CREDENTIALS="/vetiver/app/service-auth.json"
```

## How to Reproduce
### Option A — Run Entire Pipeline
Clone the repo:
```bash
git clone https://github.com/your-repo/proj-01-dank-corgi.git
cd proj-01-dank-corgi
```

Activate environment and install deps:
```bash
pip install -r vetiver_requirements.txt
```

Render the Quarto script:
```bash
quarto render api.qmd
```

Build and run Docker:
```bash
docker build -t prediabetes-model-3 .
docker run -p 2222:2222 prediabetes-model-3
```

### Option B — Use Live API (no build needed)

Simply send JSON data to:
```bash
POST http://appliedml.infosci.cornell.edu:2222/predict
```

See /docs for schema and examples.

Example Prediction Schema
```JSON
{
  "age_years": 43,
  "sex": "Female",
  "race_eth": "Non-Hispanic White",
  "sleep_hours_weekdays": 7,
  "sedentary_minutes": 360,
  "drinks_per_day_12mo": 1,
  "smoked_100_cigs_ans": "No",
  "smoke_now_ans": "Not at all",
  "tried_lose_weight_past_year_ans": "Yes",
  "regular_periods_12mo_ans": "Yes",
  "dpq_poor_appetite_ans": "No",
  "dpq_feeling_down_ans": "No",
  "ever_asthma_ans": "No",
  "anemia_treatment_3mo_ans": "No",
  "ever_chf_ans": "No",
  "ever_heart_attack_ans": "No",
  "ever_stroke_ans": "No",
  "ever_thyroid_problem_ans": "No",
  "ever_liver_condition_ans": "No",
  "ever_high_bp_ans": "Yes",
  "ever_high_chol_ans": "No",
  "bmi_from_selfreport": 26.3,
  "weight_change_1y_lb": -5,
  "ever_smoker": 0,
  "current_smoker": 0
}
```

### Interpretation
Prediction	Meaning
0	Model predicts the participant is not prediabetic
1	Model predicts the participant is prediabetic



## Summary

This end-to-end pipeline demonstrates how to:
1. Train a reproducible ML model in scikit-learn
2. Deploy it using Vetiver + FastAPI + Docker
3. Serve live predictions via RESTful API
4. Document and evaluate model performance transparently

The final product is a fully functional, cloud-hosted prediction API that supports structured input and real-time inference.