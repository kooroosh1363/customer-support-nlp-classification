# NLP Classification Method Card

## Intended use
Portfolio demonstration of multi-class customer-support intent classification and confidence-aware routing using the Banking77 benchmark.

## Evaluation design
The official Banking77 test file is kept untouched until final evaluation. The official training file is split 80/20 with stratification to create train and validation sets. Model selection and escalation-threshold selection use validation only.

## Candidate models
- majority-class baseline
- TF-IDF + Multinomial Naive Bayes
- TF-IDF + class-weighted Logistic Regression
- TF-IDF + class-weighted Linear SVM

## Primary metric
Macro-F1 is the primary selection metric because there are 77 classes and performance should not be dominated by larger classes. Macro Recall is used as a tie-break.

## Confidence-aware routing
The system separates classification from automation policy. High-confidence predictions may be auto-routed; lower-confidence requests are escalated for human review.

For probabilistic classifiers, confidence is the maximum predicted class probability. For LinearSVC, confidence is the gap between the top and second decision-function scores. That SVC margin gap is a ranking confidence signal, not a calibrated probability.

The validation threshold is selected to target at least 90% accuracy among auto-routed validation examples while maximizing coverage. The exact locked threshold is then applied unchanged to the official test set.

## Error analysis
The pipeline exports up to 100 high-confidence misclassifications for manual review. This is useful for identifying semantically adjacent intents and overconfident failure modes.

## Limitations
- Banking77 is banking-specific, not a universal support taxonomy;
- lexical benchmark performance may not transfer to noisy production tickets;
- TF-IDF models do not model long-range semantic context like modern transformers;
- LinearSVC margin gaps are not calibrated probabilities;
- class labels may overlap semantically and operational taxonomies can drift;
- offline Macro-F1 does not prove CSAT, first-contact resolution, handling-time reduction, or cost savings.

## Production extensions
Production work would add calibrated probabilities, transformer embeddings or fine-tuning, multilingual support, PII redaction, taxonomy/version governance, drift monitoring, review-capacity-aware thresholds, active learning, human feedback loops, and online routing metrics.
