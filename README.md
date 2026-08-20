# Smart Energy Consumption Forecasting — Data-Driven Systems Engineering Project
This repository is the working codebase and deliverable set for **all ten sprints (0-9) + Refactoring Extensions (RE)** of the Data-Driven Systems Engineering project **Smart Energy Consumption Forecasting**.  
It implements, end to end, three Data-Driven Systems Engineering pillars:

1. **Data engineering** — real + synthetic ingestion, validation, cleaning, transformation, feature engineering, and feature store.

2. **ML engineering** — chronological train/val/test splitting, a naive baseline, four learned models (*Linear Regression*, *Random Forest*, *XGBoost*, *LightGBM*), real `MLflow` experiment tracking and model registry, real `SHAP` explainability.

3. **App engineering** — a `FastAPI` prediction service and a **real-time** `Streamlit` dashboard (live weather + simulated live meter stream, `River` `ADWIN` + `Evidently` drift panels, CSV upload only as an offline fallback) with rule-based optimisation suggestions, plus fully-automated MLOps monitoring and retraining.

<br>

## Quick start

```bash
# ---------------------------------------------------------------------------------------------------------------------- #
# Manual step-by-step execution                                                                                          #
# ---------------------------------------------------------------------------------------------------------------------- #

# Setup the virtual environment
# a) For a Unix-like shell (Linux/macOS/bash), use
py -3.13 -m venv .venv && source .venv/bin/activate

# b) For Windows PowerShell, use
py -3.13 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1

# Install the required libraries
pip install -r requirements.txt

# Sprints 1-3 → Ingest (real-data, with synthetic fallback) + validate + clean + features
python3 scripts/run_pipeline.py

# Sprints 4-5 → Train and register 4 models (Linear/RF/XGBoost/LightGBM)
# Compare them against the naive baseline (real MLflow tracking + registry)
python3 scripts/train_models.py

# Promote the champion to Production
python3 scripts/promote_champion.py                               # Auto-selects lowest-MAE champion
python3 scripts/promote_champion.py --dry-run                     # Report only
python3 scripts/promote_champion.py --model xgboost --version 2   # Explicit override

# Sprint 6 → Serve predictions (needs a Production-stage model — run promote_champion.py first)
uvicorn src.app.api:app --reload --port 8000

# Sprint 7 → REAL-TIME dashboard (live weather + simulated live meter,
# ADWIN + Evidently drift panels; falls back to CSV upload only if offline)
# In another terminal, activate the virtual environment first. Then
# a) For a Unix-like shell (Linux/macOS/bash), use
API_BASE_URL=http://localhost:8000 streamlit run src/app/dashboard.py

# b) For Windows PowerShell, use
$env:API_BASE_URL="http://localhost:8000"
streamlit run src/app/dashboard.py

# Sprint 8 → Drift check against fresh ONLINE data,
# with fully automated retrain + promotion if a retrain is indicated
python3 scripts/run_drift_check.py
python3 scripts/run_drift_check.py --no-auto-retrain   # Report only, never retrain


# ---------------------------------------------------------------------------------------------------------------------- #
# Complete end-to-end demo in one process                                                                                #
# ---------------------------------------------------------------------------------------------------------------------- #

# Sprint 8 → Every step of the manual execution, in one process, with real printed numbers
python3 scripts/demo_end_to_end.py


# ---------------------------------------------------------------------------------------------------------------------- #
# Tests                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------- #

# Tests (Sprint 2 CI gate, extended through Sprint 8 + Refactoring)
pytest -q                                          # Dependency-light suite (pandas/numpy/sklearn/scipy only)
pytest -q tests/test_api.py                        # Requires fastapi+httpx
pytest -q tests/test_ml_tracking_and_registry.py   # Requires mlflow
pytest -q tests/test_mlops_monitoring.py           # Requires evidently
pytest -q tests/test_streaming_drift.py            # Requires river
pytest -q tests/test_validate.py                   # Requires great_expectations


# ---------------------------------------------------------------------------------------------------------------------- #
# Docker                                                                                                                 #
# ---------------------------------------------------------------------------------------------------------------------- #

# Run everything containerised
docker compose -f docker/docker-compose.yml up
```

<br>

## Repository layout

```
energy-forecast-project/
├── .github/workflows/ci.yml   # CI pipeline (Sprint 2, extended through Sprint 8)
├── data/
│   ├── raw/                   # Immutable ingested data (Sprint 1, real+fallback)
│   ├── processed/             # Cleaned & transformed (Sprint 3)
│   └── feature_store/         # Versioned feature tables (Sprint 3)
├── docker/                    # Dockerfile(s) + docker-compose (Sprint 2/6/7/8)
├── mlruns/                    # MLflow tracking store
├── model_registry_pickles/    # Explicit Pickle model artifacts (Change 1; replaces model_registry/)
├── reports/                   # Data-Quality and SHAP summaries
├── scripts/                   # CLI entry points / orchestration scripts
├── src/
│   ├── app/                   # FastAPI serving + Streamlit dashboard + recommendations
│   ├── common/                # Config, logging, dataset versioning helpers
│   ├── data_engineering/      # Ingest (real+fallback) → validate → clean → transform → features
│   ├── feature_store/         # Lightweight local feature store implementation
│   ├── ml/                    # Dataset split, baseline, train, evaluate, tracking, registry, explain
│   ├── mlops/                 # Drift detection + retraining trigger (Sprint 8)
│   └── streaming/             # Real-time online data client
└── tests/                     # Pytest suite (unit + integration, incl. API tests)
```