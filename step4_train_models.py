"""
STEP 4: MODEL TRAINING
=========================
INTERVIEW TALKING POINT:
"I follow a model ladder: start simple, then add complexity only if it
earns its keep.
  1. Logistic Regression -- fast, interpretable baseline. Good for
     explaining to non-technical stakeholders (Trust & Safety, Legal)
     WHY a product got flagged, since coefficients are interpretable.
  2. Random Forest -- captures nonlinear feature interactions
     (e.g. 'high discount AND low seller rating AND high-risk category'
     is a much stronger signal than any one feature alone).
  3. Gradient Boosting -- usually the strongest tabular model; in
     production I'd swap this for XGBoost/LightGBM since they're faster
     and have scale_pos_weight for class imbalance, but I'm using
     sklearn's GradientBoostingClassifier here since it needs no
     extra install and the API is functionally equivalent for this demo.

I handle the ~15% positive-class imbalance with class_weight='balanced'
on the linear/forest models, so false negatives (missed counterfeits)
and false positives (wrongly flagged genuine items) are penalized
according to their actual class frequency."

WHAT THIS STEP DOES:
- Loads Step 3's train split
- Trains 3 models: logistic regression, random forest, gradient boosting
- Saves trained models (pickled) for Step 5 to evaluate

Run: python step4_train_models.py
"""

import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42


if __name__ == "__main__":
    print("STEP 4: Training models (logistic regression -> random forest -> gradient boosting)")

    train_df = pd.read_csv("data/step3_train.csv")
    X_train = train_df.drop(columns=["is_counterfeit"])
    y_train = train_df["is_counterfeit"]
    print(f"Training on {len(X_train)} rows, {X_train.shape[1]} features")

    # ---- Model 1: Logistic Regression (needs scaled features) ----
    print("\n[1/3] Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    log_reg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    log_reg.fit(X_train_scaled, y_train)
    print("  done.")

    # ---- Model 2: Random Forest (no scaling needed) ----
    print("[2/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("  done.")

    # ---- Model 3: Gradient Boosting ----
    print("[3/3] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)
    print("  done.")

    # ---- Persist everything Step 5 needs ----
    artifacts = {
        "logistic_regression": {"model": log_reg, "scaler": scaler},
        "random_forest": {"model": rf, "scaler": None},
        "gradient_boosting": {"model": gb, "scaler": None},
    }
    with open("data/step4_models.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("\nSaved -> data/step4_models.pkl (3 trained models)")
