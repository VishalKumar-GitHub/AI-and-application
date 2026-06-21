"""
STEP 6: BUSINESS-DRIVEN THRESHOLD SELECTION
==============================================
INTERVIEW TALKING POINT -- THIS IS THE KEY DIFFERENTIATOR FOR A SENIOR ANSWER:
"A junior answer stops at 'pick the model with the best F1.' A senior
answer ties the threshold choice back to the GUARDRAIL METRICS defined
at the start of the design: genuine-product upload rate and seller churn
rate must stay flat. That means whatever I auto-remove must be done with
HIGH PRECISION -- I cannot afford to wrongly ban good sellers.

So instead of using the default 0.5 cutoff, I scan the precision-recall
curve and find the LOWEST threshold that still satisfies a minimum
precision target (e.g. 0.85). That maximizes recall -- catching as many
real counterfeits as possible -- without violating the guardrail.

This threshold is a STARTING POINT. The actual rollout threshold gets
validated further via the online A/B experiment in Step 8."

WHAT THIS STEP DOES:
- Loads the best model from Step 5 + validation set
- Computes the precision-recall curve
- Finds the lowest threshold meeting a minimum precision guardrail
- Saves the chosen threshold for Step 7 to use

Run: python step6_threshold_selection.py
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve

MIN_PRECISION_GUARDRAIL = 0.85  # business constraint: protect genuine sellers


def pick_operating_threshold(y_true, y_proba, min_precision: float):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]  # align lengths with thresholds

    valid = precisions >= min_precision
    if not valid.any():
        return None, None, None

    best_idx = np.argmax(recalls[valid])
    return thresholds[valid][best_idx], precisions[valid][best_idx], recalls[valid][best_idx]


if __name__ == "__main__":
    print("STEP 6: Business-driven threshold selection")
    print(f"Guardrail: minimum precision = {MIN_PRECISION_GUARDRAIL}\n")

    val_df = pd.read_csv("data/step3_val.csv")
    X_val = val_df.drop(columns=["is_counterfeit"])
    y_val = val_df["is_counterfeit"]

    with open("data/step4_models.pkl", "rb") as f:
        artifacts = pickle.load(f)
    with open("data/step5_best_model_name.txt") as f:
        best_model_name = f.read().strip()

    print(f"Using best model from Step 5: {best_model_name}")
    model = artifacts[best_model_name]["model"]
    scaler = artifacts[best_model_name]["scaler"]

    X_eval = X_val if scaler is None else scaler.transform(X_val)
    y_proba = model.predict_proba(X_eval)[:, 1]

    threshold, precision, recall = pick_operating_threshold(y_val, y_proba, MIN_PRECISION_GUARDRAIL)

    if threshold is None:
        print("\nNo threshold meets the precision guardrail with this model. "
              "Would need a better model or more features before launch.")
    else:
        print(f"\nChosen threshold: {threshold:.4f}")
        print(f"  -> precision: {precision:.4f} (meets >= {MIN_PRECISION_GUARDRAIL} guardrail)")
        print(f"  -> recall:    {recall:.4f} (this is the % of counterfeits we'll catch)")

        with open("data/step6_threshold.txt", "w") as f:
            f.write(str(threshold))
        print("\nSaved -> data/step6_threshold.txt")
