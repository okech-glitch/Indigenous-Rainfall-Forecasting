from __future__ import annotations
import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from .preprocess import build_preprocess_pipeline, TARGET_COL, ID_COL


def train_model(train_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(train_path)

    preprocessor, feature_cols = build_preprocess_pipeline(df)
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Encode string labels to integers for XGBoost
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        num_class=len(label_encoder.classes_),
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", model)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = []
    oof_trues = []

    for train_idx, val_idx in skf.split(X, y_enc):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        # Ensure preds are class indices, not probabilities
        if hasattr(preds, 'shape') and len(preds.shape) > 1:
            preds = preds.argmax(axis=1)
        oof_preds.extend(preds.tolist())
        oof_trues.extend(y_val.tolist())

    f1 = f1_score(oof_trues, oof_preds, average="macro")
    print(f"CV Macro F1: {f1:.4f}")
    print(
        classification_report(
            oof_trues, oof_preds, target_names=label_encoder.classes_
        )
    )

    pipeline.fit(X, y_enc)
    artifact = {"pipeline": pipeline, "label_encoder": label_encoder}
    joblib.dump(artifact, os.path.join(output_dir, "final_model.joblib"))


def predict_and_submit(predict_path: str, model_path: str, sample_submission: str, submission_path: str) -> None:
    df_test = pd.read_csv(predict_path)
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]

    feature_cols = [c for c in df_test.columns if c not in [ID_COL]]
    pred_int = pipeline.predict(df_test[feature_cols])
    # Ensure pred_int are class indices, not probabilities
    if hasattr(pred_int, 'shape') and len(pred_int.shape) > 1:
        pred_int = pred_int.argmax(axis=1)
    preds = label_encoder.inverse_transform(pred_int.astype(int))

    sub = pd.read_csv(sample_submission)
    sub["Target"] = preds
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    sub.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--predict_path")
    parser.add_argument("--model_path", default="results/final_model.joblib")
    parser.add_argument("--sample_submission", default="data/SampleSubmission.csv")
    parser.add_argument("--submission_path", default="results/submission.csv")
    args = parser.parse_args()

    if args.train_path:
        train_model(args.train_path, args.output_dir)
    if args.predict_path:
        predict_and_submit(
            predict_path=args.predict_path,
            model_path=args.model_path,
            sample_submission=args.sample_submission,
            submission_path=args.submission_path,
        )


if __name__ == "__main__":
    main()
