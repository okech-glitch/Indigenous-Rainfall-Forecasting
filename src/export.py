from __future__ import annotations
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_to_onnx(model_path: str, onnx_path: str, example_csv: str | None = None) -> None:
    pipeline = joblib.load(model_path)
    if example_csv and os.path.exists(example_csv):
        df = pd.read_csv(example_csv)
        n_features = df.drop(columns=[c for c in ["Target", "ID"] if c in df.columns]).shape[1]
    else:
        # Fallback: must provide a shape; use 100 as a safe upper bound
        n_features = 100
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model to {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--example_csv")
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.onnx_path, args.example_csv)


if __name__ == "__main__":
    main()
