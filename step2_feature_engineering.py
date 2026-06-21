"""
STEP 2: FEATURE ENGINEERING
=============================
INTERVIEW TALKING POINT:
"Now that I have raw entity-level data, I need to prepare it for modeling.
The categorical feature here is 'category' -- I one-hot encode it. I keep
product_id and seller_id around for now but EXCLUDE them from the actual
model input matrix, because as raw IDs they're meaningless to a tree or
linear model. In a neural network variant, seller_id and product_id would
instead go through an nn.Embedding layer to let the model learn latent
representations of seller/product risk -- I'll mention that as a possible
v2 improvement."

WHAT THIS STEP DOES:
- Loads Step 1's output
- One-hot encodes the 'category' column
- Defines the final feature list used by the model
- Saves the feature-engineered dataframe to disk for Step 3

Run: python step2_feature_engineering.py
"""

import pandas as pd

CATEGORICAL_FEATURES = ["category"]

# columns NOT fed into the model directly (IDs / target)
ID_AND_TARGET_COLS = ["product_id", "seller_id", "is_counterfeit"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix="cat")
    return df


if __name__ == "__main__":
    print("STEP 2: Feature engineering")

    df = pd.read_csv("data/step1_raw_products.csv")
    print(f"Loaded raw data: {df.shape}")

    df_features = engineer_features(df)

    feature_cols = [c for c in df_features.columns if c not in ID_AND_TARGET_COLS]
    print(f"\nModel-ready feature count: {len(feature_cols)}")
    print("Features:")
    for c in feature_cols:
        print(f"  - {c}")

    df_features.to_csv("data/step2_features.csv", index=False)
    print("\nSaved -> data/step2_features.csv")
