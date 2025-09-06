# Indigenous Rainfall Forecasting

Predict rainfall type (HEAVY, MODERATE, SMALL, NORAIN) in the next 12–24 hours using Indigenous Ecological Indicators (IEIs) from Ghanaian farmers (Pra River Basin). Includes explainability and ONNX export. Aligns with RAIL challenge.

## Quickstart

1) Install

```bash
python -m venv .venv
# PowerShell
. .venv/Scripts/Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2) Prepare data

Place `train.csv`, `test.csv`, and `SampleSubmission.csv` under `data/`.

3) Train and evaluate

```bash
python -m src.model --train_path data/train.csv --output_dir results
```

4) Explainability

```bash
python -m src.explainability --train_path data/train.csv --model_path results/final_model.joblib --output_dir results/explainability
```

5) Export to ONNX

```bash
python -m src.export --model_path results/final_model.joblib --onnx_path results/model.onnx
```

6) Predict for submission

```bash
python -m src.model --predict_path data/test.csv --model_path results/final_model.joblib --submission_path results/submission.csv --sample_submission data/SampleSubmission.csv
```

## Performance

- **CV Macro F1 Score**: 0.7528
- **Model**: XGBoost with preprocessing pipeline
- **Features**: Indigenous Ecological Indicators (IEIs)

## CI

GitHub Actions runs flake8 and pytest on push/PR.

## License

MIT
