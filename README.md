# Counterfeit Fraud Detection — Step-by-Step ML System Design

Companion code for the "ML System Design: Counterfeit Goods Fraud Detection"
walkthrough. Each step is a standalone, runnable script. Run them in order —
each one reads the previous step's output from `data/` and writes its own.

## Why split into separate files?

For an interview, this lets you run and narrate ONE concept at a time instead
of dumping a giant script. Each file's docstring contains the **talking
point** you'd say out loud at that stage of a system design interview.

## How to run

```bash
pip install pandas numpy scikit-learn

python step1_simulate_data.py              # ingest / simulate raw data
python step2_feature_engineering.py        # build entity features
python step3_train_test_split.py           # stratified 70/15/15 split
python step4_train_models.py               # train logreg -> RF -> GBM
python step5_evaluate_models.py            # compare on validation set
python step6_threshold_selection.py        # guardrail-driven threshold
python step7_risk_tiering_and_test_eval.py # final test-set evaluation
```

## The story arc (what to say at each step)

| Step | File | What you're demonstrating |
|------|------|---------------------------|
| 1 | `step1_simulate_data.py` | Understanding of what real data sources would feed this (catalog DB, seller DB, returns logs, image-similarity service) |
| 2 | `step2_feature_engineering.py` | Translating the 4 entity groups (product, seller, category, context) from the design into actual features |
| 3 | `step3_train_test_split.py` | Awareness of class imbalance + correct stratified splitting + train/val/test discipline |
| 4 | `step4_train_models.py` | Model ladder reasoning: start simple (interpretable baseline), add complexity only if it earns it |
| 5 | `step5_evaluate_models.py` | Using the RIGHT metrics for imbalanced fraud problems (not accuracy) |
| 6 | `step6_threshold_selection.py` | **The senior-level differentiator** — tying the threshold choice back to business guardrail metrics, not just optimizing F1 |
| 7 | `step7_risk_tiering_and_test_eval.py` | Translating model output into the actual production architecture (auto-remove / manual-review / keep) + disciplined held-out test evaluation |

## What's intentionally left as "discussion, not code"

These are things to talk through verbally in the interview rather than code,
since they require live infrastructure:

- **Online A/B experimentation** — control (existing methodology) vs.
  treatment (new model), null/alternative hypothesis, significance level,
  p-value, Type I/II errors, before scaling to 100% of traffic.
- **Production monitoring** — feature drift (e.g. PSI), label drift,
  scheduled retraining, alerting if precision on audited "keep" samples
  degrades over time.
- **Feedback loop** — manual review outcomes + auto-removed high-risk items
  flow back into the next training run (mentioned in `step4`'s docstring).
- **Embeddings for IDs** — in a neural net variant, `seller_id` /
  `product_id` would go through an `nn.Embedding` layer instead of being
  dropped, to let the model learn latent seller/product risk representations.
