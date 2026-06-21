"""
STEP 5: MODEL EVALUATION
===========================
INTERVIEW TALKING POINT:
"I evaluate on the VALIDATION set (never the test set yet -- that's
reserved for the final unbiased check). For a fraud problem, accuracy is
a misleading metric because of class imbalance -- a model that predicts
'genuine' for everything would already be ~85% accurate while catching
zero counterfeits. So I look at:
  - Precision: of products I flag as counterfeit, how many actually are?
    (directly protects the guardrail metric -- genuine seller upload rate)
  - Recall: of all actual counterfeits, how many did I catch?
    (directly drives the North Star metric -- counterfeit reduction)
  - F1: harmonic mean, useful single-number summary
  - ROC-AUC: threshold-independent ranking quality, good for comparing
    models before I've picked an operating threshold

I pick the model with the best ROC-AUC as the candidate to carry forward
into threshold tuning."

WHAT THIS STEP DOES:
- Loads Step 4's trained models + Step 3's validation split
- Scores each model on the validation set
- Prints a comparison table
- Saves the comparison + identifies the best model for Step 6

Run: python step5_evaluate_models.py
"""

import pickle
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


if __name__ == "__main__":
    print("STEP 5: Evaluating models on validation set")

    val_df = pd.read_csv("data/step3_val.csv")
    X_val = val_df.drop(columns=["is_counterfeit"])
    y_val = val_df["is_counterfeit"]

    with open("data/step4_models.pkl", "rb") as f:
        artifacts = pickle.load(f)

    rows = []
    for name, art in artifacts.items():
        model, scaler = art["model"], art["scaler"]
        X_eval = X_val if scaler is None else scaler.transform(X_val)
        y_proba = model.predict_proba(X_eval)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        rows.append({
            "model": name,
            "precision": round(precision_score(y_val, y_pred), 4),
            "recall": round(recall_score(y_val, y_pred), 4),
            "f1": round(f1_score(y_val, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_val, y_proba), 4),
        })

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    print("\nValidation set results (threshold=0.5):")
    print(results.to_string(index=False))

    best_model_name = results.iloc[0]["model"]
    print(f"\nBest model by ROC-AUC: {best_model_name}")

    results.to_csv("data/step5_results.csv", index=False)
    with open("data/step5_best_model_name.txt", "w") as f:
        f.write(best_model_name)

    print("\nSaved -> data/step5_results.csv, data/step5_best_model_name.txt")
