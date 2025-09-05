import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_SUBDIR = THIS_FILE.parents[1]
sys.path.insert(0, str(REPO_SUBDIR))

import os  # noqa: E402
import pandas as pd  # noqa: E402
from src.model import train_model, predict_and_submit  # noqa: E402


def test_train_and_predict_tmp(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    sample_sub_csv = tmp_path / "SampleSubmission.csv"
    results_dir = tmp_path / "results"

    df_train = pd.DataFrame({
        "ID": ["A", "B", "C", "D"],
        "cloud": ["LOW", "HIGH", "MED", "LOW"],
        "wind": ["CALM", "BREEZY", "CALM", "BREEZY"],
        "rain_mm": [0.0, 12.3, 3.2, 0.5],
        "Target": ["NORAIN", "SMALL", "SMALL", "NORAIN"],
    })
    df_train.to_csv(train_csv, index=False)

    df_test = pd.DataFrame({
        "ID": ["X", "Y"],
        "cloud": ["LOW", "HIGH"],
        "wind": ["CALM", "BREEZY"],
        "rain_mm": [0.0, 1.2],
    })
    df_test.to_csv(test_csv, index=False)

    pd.DataFrame({"ID": ["X", "Y"], "Target": ["NORAIN", "NORAIN"]}).to_csv(sample_sub_csv, index=False)

    train_model(str(train_csv), str(results_dir))
    model_path = results_dir / "final_model.joblib"
    assert model_path.exists()

    submission_path = results_dir / "submission.csv"
    predict_and_submit(str(test_csv), str(model_path), str(sample_sub_csv), str(submission_path))
    assert submission_path.exists()
