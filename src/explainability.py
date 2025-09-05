from __future__ import annotations
import argparse
import os
import joblib
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

from .preprocess import TARGET_COL, ID_COL, build_preprocess_pipeline


def run_shap(train_path: str, model_path: str, output_dir: str, num_samples: int = 512) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(train_path)
    pipeline = joblib.load(model_path)

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    X = df[feature_cols]

    try:
        explainer = shap.Explainer(pipeline.named_steps["clf"], feature_names=None)
        shap_values = explainer(pipeline.named_steps["pre"].fit_transform(X)[:num_samples])
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"SHAP fallback KernelExplainer due to: {e}")
        background = shap.sample(X, min(200, len(X)))
        kexpl = shap.KernelExplainer(pipeline.predict_proba, background)
        sv = kexpl.shap_values(shap.sample(X, min(num_samples, len(X))))
        shap.summary_plot(sv, shap.sample(X, min(num_samples, len(X))), show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_kernel.png"), dpi=200)
        plt.close()


def run_lime(train_path: str, model_path: str, output_dir: str, num_samples: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(train_path)
    pipeline = joblib.load(model_path)
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    X = df[feature_cols]

    explainer = LimeTabularExplainer(
        training_data=pipeline.named_steps["pre"].fit_transform(X).toarray(),
        feature_names=[str(i) for i in range(pipeline.named_steps["pre"].transform(X).shape[1])],
        discretize_continuous=True,
        mode="classification",
    )

    for i in range(min(num_samples, len(X))):
        instance = X.iloc[i]
        exp = explainer.explain_instance(
            data_row=pipeline.named_steps["pre"].transform(instance.to_frame().T).toarray()[0],
            predict_fn=pipeline.named_steps["clf"].predict_proba,
            num_features=10,
        )
        fig = exp.as_pyplot_figure()
        fig.savefig(os.path.join(output_dir, f"lime_{i}.png"), dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    run_shap(args.train_path, args.model_path, args.output_dir)
    run_lime(args.train_path, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
