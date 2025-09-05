# Rainfall Forecasting Using Indigenous Ecological Indicators

## Problem Statement
Predicting rainfall in rural Ghana remains challenging due to sparse, delayed, or inaccessible weather infrastructure. Indigenous ecological indicators (IEIs)—such as cloud formations, sun/moon cues, wind, heat, and animal/tree behaviors—provide hyper-local, time-sensitive signals. This project digitizes IEIs to forecast rainfall type (heavy, moderate, small, no rain) within 12–24 hours, supporting inclusive, community-informed climate resilience.

## Objectives
- Classify rainfall type for 12–24 hour horizons using IEIs.
- Validate and complement indigenous methods with ML.
- Provide transparent predictions with explainability (SHAP/LIME).
- Export models to ONNX/TFLite.
- Optimize F1 score; ensure reproducibility and GitHub CI.

## Stakeholders
- Ghanaian farmers (Pra River Basin)
- RAIL, French Embassy, AI4D Africa, Zindi community

## Scope
- End-to-end classification pipeline (preprocess → train → export → explainability → submission)
- GitHub integration with CI (pytest, flake8)
- Lightweight models suitable for low-resource settings

## Data Description
- train.csv: IEIs, target rainfall type, rainfall measurements, timeframe
- test.csv: Same features without targets
- SampleSubmission.csv: ID, Target
- Starter Notebook: Indigenous_Weather_Forecasting-Starter_Notebook.ipynb

Typical variables: IEIs (categorical), rainfall amounts (numeric), timeframe (12/24 hr), accuracy/quality flags.

## Success Metrics
- Primary: Macro F1 score on holdout / leaderboard
- Secondary: Explainability quality and clarity; model portability

## Constraints
- Limited connectivity; prefer small, fast models and minimal dependencies
- Ethical AI: transparency, respect for indigenous knowledge, data privacy

## Deliverables
- Trained model (ONNX or TFLite)
- Results: submission.csv
- Docs: README, PRD, explainability report with visuals
- Code: modular `src/` with tests and CI

## Timeline (2–3 weeks)
- Week 1: Data prep/EDA, baseline model, CI
- Week 2: Tuning, explainability, export
- Week 3: Validation, documentation, submission
