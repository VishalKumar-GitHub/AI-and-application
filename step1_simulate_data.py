"""
STEP 1: DATA INGESTION (simulated)
===================================
INTERVIEW TALKING POINT:
"In production, this data wouldn't be simulated -- it would come from
joining the product catalog DB, seller DB, returns/refund event logs,
a review-sentiment microservice, and an image-similarity service that
compares product photos against a known-counterfeit embedding index.
Here I simulate it so the pipeline runs end-to-end and the data has
realistic, correlated signal."

WHAT THIS STEP DOES:
- Creates ~20K synthetic products (standing in for the platform's ~60M)
- Assigns each product a seller, category, and a *latent* (hidden) risk
  level that we use only to generate realistic correlated features + label
- Saves the raw dataframe to disk so Step 2 can pick it up

Run: python step1_simulate_data.py
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def simulate_dataset(n_products: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    # ---- latent (hidden) risk, used only to generate realistic data ----
    base_risk = rng.beta(a=2, b=8, size=n_products)  # most products low risk

    categories = rng.choice(
        ["electronics", "apparel", "beauty", "accessories", "home_goods"],
        size=n_products, p=[0.25, 0.25, 0.2, 0.15, 0.15]
    )
    category_susceptibility = {
        "electronics": 0.7, "apparel": 0.65, "beauty": 0.5,
        "accessories": 0.6, "home_goods": 0.2,
    }
    cat_risk_boost = np.array([category_susceptibility[c] for c in categories]) * 0.3

    seller_id = rng.integers(0, 4000, size=n_products)
    bad_seller_ids = set(rng.choice(4000, size=150, replace=False))
    seller_is_bad = np.array([sid in bad_seller_ids for sid in seller_id])
    seller_risk_boost = seller_is_bad.astype(float) * 0.35

    latent_risk = np.clip(base_risk + cat_risk_boost + seller_risk_boost, 0, 1)
    noise = rng.normal(0, 0.08, size=n_products)
    label = ((latent_risk + noise) > 0.55).astype(int)

    df = pd.DataFrame({
        "product_id": [f"P{i:07d}" for i in range(n_products)],
        "seller_id": seller_id,
        "category": categories,
        "is_counterfeit": label,  # TARGET -- comes from manual review in reality
    })

    # ---- PRODUCT features ----
    df["listed_price"] = np.round(rng.gamma(2, 40, n_products) + 5, 2)
    df["discount_pct"] = np.clip(rng.normal(15 + latent_risk * 25, 10, n_products), 0, 90)
    df["return_rate"] = np.clip(rng.normal(0.05 + latent_risk * 0.15, 0.04, n_products), 0, 1)
    df["refund_rate"] = np.clip(rng.normal(0.03 + latent_risk * 0.12, 0.03, n_products), 0, 1)
    df["counterfeit_return_ratio"] = np.clip(rng.normal(latent_risk * 0.4, 0.05, n_products), 0, 1)
    df["image_quality_score"] = np.clip(rng.normal(0.75 - latent_risk * 0.2, 0.15, n_products), 0, 1)
    df["image_similarity_to_known_counterfeit"] = np.clip(rng.normal(latent_risk * 0.6, 0.15, n_products), 0, 1)
    df["brand_name_similarity_score"] = np.clip(rng.normal(latent_risk * 0.5, 0.2, n_products), 0, 1)
    df["has_unauthorized_celebrity_image"] = rng.binomial(1, latent_risk * 0.1)
    df["review_sentiment_counterfeit_signal"] = np.clip(rng.normal(latent_risk * 0.5, 0.15, n_products), 0, 1)

    # ---- SELLER features ----
    seller_features = pd.DataFrame({"seller_id": np.arange(4000)})
    seller_features["seller_is_bad_actor"] = seller_features["seller_id"].isin(bad_seller_ids).astype(int)
    seller_features["seller_tenure_days"] = rng.integers(10, 2000, 4000)
    seller_features["seller_num_listings"] = rng.integers(1, 5000, 4000)
    seller_features["seller_rating"] = np.clip(
        rng.normal(4.3 - seller_features["seller_is_bad_actor"] * 1.2, 0.4, 4000), 1, 5
    )
    seller_features["seller_violation_count"] = (
        seller_features["seller_is_bad_actor"] * rng.integers(1, 8, 4000)
    )
    df = df.merge(seller_features, on="seller_id", how="left")

    # ---- CATEGORY features ----
    cat_features = pd.DataFrame({
        "category": list(category_susceptibility.keys()),
        "category_susceptibility": list(category_susceptibility.values()),
        "category_avg_value": [180, 60, 35, 50, 90],
        "category_replication_ease": [0.7, 0.6, 0.5, 0.55, 0.15],
    })
    df = df.merge(cat_features, on="category", how="left")

    # ---- CONTEXT features ----
    df["is_peak_sale_season"] = rng.binomial(1, 0.2, n_products)

    return df


if __name__ == "__main__":
    print("STEP 1: Simulating product-level dataset (60M products -> sampled to 20K)")
    df = simulate_dataset(n_products=20_000)

    print(f"Shape: {df.shape}")
    print(f"Counterfeit rate (label=1): {df['is_counterfeit'].mean():.2%}")
    print("\nSample rows:")
    print(df.head(3).T)

    df.to_csv("data/step1_raw_products.csv", index=False)
    print("\nSaved -> data/step1_raw_products.csv")
