# NLP Classification Method Card

## Intended use
Portfolio demonstration of multi-class customer-support intent classification and confidence-aware routing using the Banking77 benchmark.

## Evaluation design
The official Banking77 test file is kept untouched until final evaluation. The official training file is split 80/20 with stratification to create train and validation sets. Model selection and escalation-threshold selection use validation only.

Exact normalized-text overlap is audited across train/validation and train/test to catch obvious leakage. This check does not prove semantic independence between splits, but it detects literal duplicate leakage.

## Candidate models
- majority-class baseline
- TF-IDF + Multinomial Naive Bayes
- TF-IDF + class-weighted Logistic Regression
- TF-IDF + class-weighted Linear SVM

## Primary metric
Macro-F1 is the primary selection metric because there are 77 classes and performance should not be dominated by larger classes. Macro Recall is used as a tie-break.

## Confidence-aware routing
The system separates classification from automation policy. High-confidence predictions may be auto-routed; lower-confidence requests are escalated for human review.

For probabilistic classifiers, confidence is the maximum predicted class probability. For LinearSVC, confidence is the **sorted gap between the highest and second-highest decision-function scores**. That SVC margin gap is a ranking confidence signal, not a calibrated probability.

The validation routing threshold is chosen by evaluating **every distinct validation confidence value**. Among thresholds achieving at least 90% validation accuracy for auto-routed examples, the policy maximizes coverage, then prefers higher accuracy, then the lower threshold. The locked threshold is applied unchanged to the official test set.

## Error analysis
The pipeline exports up to 100 overall high-confidence misclassifications and a separate file containing auto-routed errors. The latter is especially important because those are failures the automation policy would have sent without human review.

## Limitations
- Banking77 is banking-specific, not a universal support taxonomy;
- lexical benchmark performance may not transfer to noisy production tickets;
- TF-IDF models do not model long-range semantic context like modern transformers;
- LinearSVC margin gaps are not calibrated probabilities;
- class labels may overlap semantically and operational taxonomies can drift;
- a 90% validation routing target is an illustrative policy, not an externally validated service-level requirement;
- offline Macro-F1 does not prove CSAT, first-contact resolution, handling-time reduction, or cost savings.

## Production extensions
Production work would add calibrated probabilities, transformer embeddings or fine-tuning, multilingual support, PII redaction, taxonomy/version governance, drift monitoring, review-capacity-aware thresholds, active learning, human feedback loops, and online routing metrics.
