"""
STEP 7: RISK TIERING + FINAL TEST EVALUATION
===============================================
INTERVIEW TALKING POINT:
"This maps directly to the architecture diagram: the model doesn't make
a binary keep/remove decision -- it assigns each product to a risk tier:
  - auto_remove   (very high risk, e.g. proba >= 0.85): pulled immediately
  - manual_review (medium-high risk, e.g. 0.5 <= proba < 0.85): queued
    for a human reviewer, since the cost of a wrong auto-removal is high
  - keep          (low risk, proba < 0.5): stays live, but a random
    sample is still periodically audited to catch label drift

This is the FIRST and ONLY time I touch the test set -- giving an
unbiased read of how the system would perform in production, since val
was already used for model selection and threshold tuning."

WHAT THIS STEP DOES:
- Loads the best model + chosen threshold from Steps 5 & 6
- Scores the held-out TEST set (never touched until now)
- Assigns risk tiers (auto_remove / manual_review / keep)
- Prints the final classification report + confusion matrix

Run: python step7_risk_tiering_and_test_eval.py
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

HIGH_RISK_THRESHOLD = 0.85   # auto-remove cutoff
MEDIUM_RISK_THRESHOLD = 0.5  # manual-review cutoff


def assign_risk_tier(proba: np.ndarray) -> np.ndarray:
    return np.where(
        proba >= HIGH_RISK_THRESHOLD, "auto_remove",
        np.where(proba >= MEDIUM_RISK_THRESHOLD, "manual_review", "keep")
    )


if __name__ == "__main__":
    print("STEP 7: Risk tiering + final test set evaluation")

    test_df = pd.read_csv("data/step3_test.csv")
    X_test = test_df.drop(columns=["is_counterfeit"])
    y_test = test_df["is_counterfeit"]

    with open("data/step4_models.pkl", "rb") as f:
        artifacts = pickle.load(f)
    with open("data/step5_best_model_name.txt") as f:
        best_model_name = f.read().strip()
    with open("data/step6_threshold.txt") as f:
        operating_threshold = float(f.read().strip())

    print(f"Model: {best_model_name} | Operating threshold: {operating_threshold:.4f}\n")

    model = artifacts[best_model_name]["model"]
    scaler = artifacts[best_model_name]["scaler"]
    X_eval = X_test if scaler is None else scaler.transform(X_test)
    test_proba = model.predict_proba(X_eval)[:, 1]

    # --- Risk tiering (matches the architecture diagram) ---
    tiers = assign_risk_tier(test_proba)
    print("Risk tier distribution on test set:")
    print(pd.Series(tiers).value_counts().to_string())

    # --- Final unbiased metrics using the guardrail-driven threshold ---
    test_pred_at_threshold = (test_proba >= operating_threshold).astype(int)
    print(f"\nClassification report at operating threshold ({operating_threshold:.4f}):")
    print(classification_report(y_test, test_pred_at_threshold, target_names=["genuine", "counterfeit"]))

    print("Confusion matrix:")
    cm = confusion_matrix(y_test, test_pred_at_threshold)
    print(pd.DataFrame(cm, index=["actual_genuine", "actual_counterfeit"],
                        columns=["pred_genuine", "pred_counterfeit"]))

    result_df = test_df.copy()
    result_df["counterfeit_proba"] = test_proba
    result_df["risk_tier"] = tiers
    result_df.to_csv("data/step7_final_scored_test.csv", index=False)
    print("\nSaved -> data/step7_final_scored_test.csv")
