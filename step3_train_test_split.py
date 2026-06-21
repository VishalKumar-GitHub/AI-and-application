"""
STEP 3: TRAIN / VALIDATION / TEST SPLIT
==========================================
INTERVIEW TALKING POINT:
"The label is imbalanced -- only ~15% of products are counterfeit, which
mirrors real platforms where counterfeit incidence is a minority class.
I use STRATIFIED splitting so train/val/test all preserve that same class
ratio. I use a 70/15/15 split: train for fitting, val for model selection
and threshold tuning, and a held-out test set I only touch once at the
very end to report unbiased final metrics."

WHAT THIS STEP DOES:
- Loads Step 2's feature-engineered data
- Splits into train (70%) / val (15%) / test (15%), stratified by label
- Saves each split to disk for Step 4

Run: python step3_train_test_split.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ID_AND_TARGET_COLS = ["product_id", "seller_id", "is_counterfeit"]


if __name__ == "__main__":
    print("STEP 3: Train / validation / test split (stratified, 70/15/15)")

    df = pd.read_csv("data/step2_features.csv")
    print(f"Loaded feature data: {df.shape}")

    feature_cols = [c for c in df.columns if c not in ID_AND_TARGET_COLS]
    X = df[feature_cols]
    y = df["is_counterfeit"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"\nTrain: {len(X_train)} rows | counterfeit rate: {y_train.mean():.2%}")
    print(f"Val:   {len(X_val)} rows | counterfeit rate: {y_val.mean():.2%}")
    print(f"Test:  {len(X_test)} rows | counterfeit rate: {y_test.mean():.2%}")

    X_train.assign(is_counterfeit=y_train).to_csv("data/step3_train.csv", index=False)
    X_val.assign(is_counterfeit=y_val).to_csv("data/step3_val.csv", index=False)
    X_test.assign(is_counterfeit=y_test).to_csv("data/step3_test.csv", index=False)

    print("\nSaved -> data/step3_train.csv, data/step3_val.csv, data/step3_test.csv")
