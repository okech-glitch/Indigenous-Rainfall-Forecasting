from __future__ import annotations
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_to_onnx(model_path: str, onnx_path: str, example_csv: str | None = None) -> None:
    # Load our artifact which contains {"pipeline": Pipeline, "label_encoder": LabelEncoder}
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
    else:
        pipeline = artifact  # assume it's directly a pipeline

    # Try converting the whole sklearn pipeline first
    try:
        if example_csv and os.path.exists(example_csv):
            df = pd.read_csv(example_csv)
            feature_df = df.drop(columns=[c for c in ["Target", "ID"] if c in df.columns], errors="ignore")
            n_features = feature_df.shape[1]
        else:
            n_features = 100
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Saved ONNX model to {onnx_path}")
        return
    except Exception as e:
        print(f"Pipeline export via skl2onnx failed: {e}")

    # Fallback: if the classifier is XGBoost, export classifier only via onnxmltools
    try:
        from xgboost import XGBClassifier  # type: ignore
        clf = getattr(pipeline, "named_steps", {}).get("clf")
        pre = getattr(pipeline, "named_steps", {}).get("pre")
        if isinstance(clf, XGBClassifier):
            try:
                import onnxmltools  # type: ignore
                # Determine transformed feature dimension using example data
                if example_csv and os.path.exists(example_csv) and pre is not None:
                    df = pd.read_csv(example_csv)
                    feature_df = df.drop(columns=[c for c in ["Target", "ID"] if c in df.columns], errors="ignore")
                    Xt = pre.fit_transform(feature_df.head(100))
                    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
                    n_features = Xt.shape[1]
                else:
                    n_features = 100
                initial_types = [("input", FloatTensorType([None, n_features]))]
                onnx_model = onnxmltools.convert_xgboost(clf, initial_types=initial_types)
                os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                print(f"Saved XGBoost classifier ONNX to {onnx_path} (preprocessing not embedded)")
                print("Note: Preprocessing is not included in this ONNX. Apply preprocessing before inference.")
                return
            except Exception as ex:
                print(f"XGBoost-only export via onnxmltools failed: {ex}")
    except Exception:
        pass

    raise RuntimeError("ONNX export failed. For XGBoost, install 'onnxmltools' or switch to a scikit-learn classifier for full pipeline export.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--example_csv")
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.onnx_path, args.example_csv)


if __name__ == "__main__":
    main()
