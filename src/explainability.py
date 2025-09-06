from __future__ import annotations
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

from .preprocess import TARGET_COL, ID_COL


def run_shap(train_path: str, model_path: str, output_dir: str, num_samples: int = 512) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(train_path)
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    X = df[feature_cols]

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    Xt = pre.fit_transform(X)
    Xt = Xt[: min(num_samples, Xt.shape[0])]
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    try:
        # Preferred path for tree-based models
        explainer = shap.TreeExplainer(clf)
        sv = explainer(Xt_dense)
        # sv can be list (old API for multiclass) or Explanation (new API)
        if isinstance(sv, list):
            for i, sv_i in enumerate(sv):
                shap.summary_plot(sv_i, Xt_dense, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"shap_summary_class_{i}.png"), dpi=200)
                plt.close()
        else:
            # New API: save one plot per class if multiclass
            if getattr(sv, "values", None) is not None and sv.values.ndim == 3:
                num_classes = sv.values.shape[2]
                for i in range(num_classes):
                    shap.plots.beeswarm(sv[..., i], show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"shap_beeswarm_class_{i}.png"), dpi=200)
                    plt.close()
            else:
                shap.plots.beeswarm(sv, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), dpi=200)
                plt.close()
    except Exception as e:
        print(f"SHAP fallback KernelExplainer due to: {e}")
        # KernelExplainer over transformed features
        background = shap.sample(pd.DataFrame(Xt_dense), min(200, Xt_dense.shape[0])).values
        kexpl = shap.KernelExplainer(lambda z: clf.predict_proba(z), background)
        sample_X = Xt_dense[: min(256, Xt_dense.shape[0])]
        sv = kexpl.shap_values(sample_X)
        if isinstance(sv, list):
            for i, sv_i in enumerate(sv):
                shap.summary_plot(sv_i, sample_X, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"shap_summary_kernel_class_{i}.png"), dpi=200)
                plt.close()
        else:
            shap.summary_plot(sv, sample_X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary_kernel.png"), dpi=200)
            plt.close()


def run_lime(train_path: str, model_path: str, output_dir: str, num_samples: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(train_path)
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    X = df[feature_cols]

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    Xt = pre.fit_transform(X)
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    explainer = LimeTabularExplainer(
        training_data=Xt_dense,
        feature_names=[str(i) for i in range(Xt_dense.shape[1])],
        discretize_continuous=True,
        mode="classification",
    )

    for i in range(min(num_samples, len(X))):
        instance = X.iloc[i]
        x_inst = pre.transform(instance.to_frame().T)
        x_inst_dense = x_inst.toarray()[0] if hasattr(x_inst, "toarray") else x_inst[0]
        exp = explainer.explain_instance(
            data_row=x_inst_dense,
            predict_fn=clf.predict_proba,
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
