import sys
from pathlib import Path

# Ensure repo subdir is on path so we can import src/*
THIS_FILE = Path(__file__).resolve()
REPO_SUBDIR = THIS_FILE.parents[1]  # Indigenous-Rainfall-Forecasting
sys.path.insert(0, str(REPO_SUBDIR))

import pandas as pd  # noqa: E402
from src.preprocess import build_preprocess_pipeline, TARGET_COL  # noqa: E402


def test_build_preprocess_pipeline_basic():
    df = pd.DataFrame({
        "ID": ["A", "B"],
        "cloud": ["LOW", "HIGH"],
        "wind": ["CALM", "BREEZY"],
        "rain_mm": [0.0, 12.3],
        TARGET_COL: ["NORAIN", "SMALL"],
    })
    pre, feats = build_preprocess_pipeline(df)
    assert len(feats) == 3  # exclude ID and Target
    X = df[feats]
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 2
