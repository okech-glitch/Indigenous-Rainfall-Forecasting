from __future__ import annotations
import argparse
from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

TARGET_COL = "Target"
ID_COL = "ID"


def build_preprocess_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).drop(
        columns=[c for c in [TARGET_COL, ID_COL] if c in df.columns], errors="ignore"
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0.3,
    )

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    return preprocessor, feature_cols


def split_train_val(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_path)
    preprocessor, feature_cols = build_preprocess_pipeline(df)
    _ = preprocessor  # constructed for downstream usage
    _ = feature_cols

    print("Preprocessing pipeline built with:")
    print({
        "n_features": len(feature_cols),
        "has_target": TARGET_COL in df.columns,
    })


if __name__ == "__main__":
    main()
