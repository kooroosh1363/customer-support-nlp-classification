# DS-08 — Customer Support NLP Classification

Portfolio-grade multi-class NLP system for customer-support intent classification and confidence-aware routing.

## What this project demonstrates

- real public support-intent benchmark (Banking77)
- official test set kept locked for final evaluation
- stratified train/validation split from the official training set
- exact-text overlap audit across splits
- 77-class customer-service intent taxonomy
- majority-class baseline
- TF-IDF + Multinomial Naive Bayes
- TF-IDF + class-weighted Logistic Regression
- TF-IDF + class-weighted Linear SVM
- Macro-F1-first model selection
- per-class precision / recall / F1
- 77×77 confusion matrix
- confidence-aware human escalation policy
- high-confidence error analysis
- model artifacts, tests, and GitHub Actions CI

## Data

The project uses **Banking77**, released by PolyAI, from the public `PolyAI-LDN/task-specific-datasets` repository. The upstream source provides separate `train.csv` and `test.csv` files plus the 77 intent categories. The source directory and files were verified before implementation. See `DATA_SOURCE.md`. 

The official test set is never used for model or threshold selection. The official train file is stratified into 80% train and 20% validation.

## Architecture

```text
Banking77 official train/test
        |
        +--> integrity + overlap audit
        |
        +--> official train
        |       -> stratified train / validation
        |
        +--> locked official test

train text
   -> TF-IDF word 1-2 grams
   -> candidate classifiers
       - majority baseline
       - MultinomialNB
       - Logistic Regression
       - LinearSVC
   -> validation Macro-F1 selection
   -> validation confidence threshold
   -> lock model + routing threshold
   -> untouched official test
   -> class metrics + confusion matrix + error analysis
```

## Why Macro-F1 is primary

This is a **77-class** routing problem. Overall accuracy or weighted averages can hide poor behavior on smaller or difficult intents. Macro-F1 gives each intent equal weight and is therefore the primary selection metric. Macro Recall is used only as a tie-breaker.

## Confidence-aware routing

Classification and automation are treated as separate decisions. The model predicts an intent, then a routing policy decides whether that prediction is confident enough for automatic routing.

The threshold is tuned **only on validation** to target at least **90% accuracy among auto-routed validation examples**, while maximizing the share of requests that can be auto-routed. Lower-confidence cases are sent to human review.

For probabilistic models, confidence is maximum predicted probability. For LinearSVC, confidence is the gap between the highest and second-highest decision scores. That margin gap is a confidence signal, **not a calibrated probability**.

## Error analysis

The pipeline exports high-confidence wrong predictions to `artifacts/error_analysis.csv`. These are especially useful because they expose dangerous failure modes: examples the model is confident about but still routes to the wrong intent.

## Claim boundary

This project demonstrates offline intent classification and confidence-aware support routing on Banking77. It does **not** prove live ticket-resolution improvement, CSAT improvement, reduced handling time, cost savings, or causal business impact.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline
```
