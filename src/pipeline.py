from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import load_dataset

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RANDOM_STATE = 42


def build_models():
    vec = dict(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True, max_features=70000)
    return {
        "majority_baseline": Pipeline([
            ("tfidf", TfidfVectorizer(**vec)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "multinomial_nb": Pipeline([
            ("tfidf", TfidfVectorizer(**vec)),
            ("model", MultinomialNB(alpha=0.5)),
        ]),
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**vec)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "linear_svc": Pipeline([
            ("tfidf", TfidfVectorizer(**vec)),
            ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def confidence_score(model: Pipeline, texts: pd.Series) -> np.ndarray:
    clf = model.named_steps["model"]
    X = model.named_steps["tfidf"].transform(texts)
    if hasattr(clf, "predict_proba"):
        return np.max(clf.predict_proba(X), axis=1)
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        return np.abs(scores)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def choose_escalation_threshold(y_true, y_pred, confidence, target_auto_accuracy=0.90):
    candidates = np.unique(np.quantile(confidence, np.linspace(0.0, 0.95, 120)))
    best = None
    for t in candidates:
        auto = confidence >= t
        if auto.sum() == 0:
            continue
        auto_acc = float(np.mean(np.asarray(y_true)[auto] == np.asarray(y_pred)[auto]))
        coverage = float(auto.mean())
        if auto_acc >= target_auto_accuracy:
            key = (-coverage, -auto_acc, float(t))
            if best is None or key < best[0]:
                best = (key, float(t), auto_acc, coverage)
    if best is None:
        idx = int(np.argmax(confidence))
        return float(confidence[idx]), float(np.asarray(y_true)[idx] == np.asarray(y_pred)[idx]), float(1 / len(confidence))
    return best[1], best[2], best[3]


def main():
    ART.mkdir(exist_ok=True)
    train, val, test, audit, classes = load_dataset(RANDOM_STATE)

    val_rows = []
    fitted = {}
    for name, model in build_models().items():
        model.fit(train["text"], train["category"])
        fitted[name] = model
        pred = model.predict(val["text"])
        row = evaluate(val["category"], pred)
        row["model"] = name
        val_rows.append(row)

    val_df = pd.DataFrame(val_rows).sort_values(["macro_f1", "macro_recall"], ascending=False).reset_index(drop=True)
    selected = str(val_df.iloc[0]["model"])
    model = fitted[selected]

    val_pred = model.predict(val["text"])
    val_conf = confidence_score(model, val["text"])
    threshold, auto_acc, coverage = choose_escalation_threshold(val["category"].to_numpy(), val_pred, val_conf)

    test_pred = model.predict(test["text"])
    test_conf = confidence_score(model, test["text"])
    test_metrics = evaluate(test["category"], test_pred)
    test_auto = test_conf >= threshold
    test_auto_accuracy = float(np.mean(test.loc[test_auto, "category"].to_numpy() == test_pred[test_auto])) if test_auto.any() else 0.0
    test_coverage = float(test_auto.mean())
    test_escalation_rate = float(1.0 - test_coverage)

    labels = classes
    cm = confusion_matrix(test["category"], test_pred, labels=labels)
    report = classification_report(test["category"], test_pred, labels=labels, output_dict=True, zero_division=0)

    errors = test.assign(predicted=test_pred, confidence=test_conf)
    errors = errors.loc[errors["category"] != errors["predicted"]].sort_values("confidence", ascending=False).head(100)

    joblib.dump({"model": model, "escalation_threshold": threshold, "confidence_type": "probability_max_or_svc_margin_gap"}, ART / "model.joblib")
    val_df.to_csv(ART / "validation_metrics.csv", index=False)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(ART / "test_confusion_matrix.csv")
    pd.DataFrame(report).T.to_csv(ART / "test_classification_report.csv")
    errors.to_csv(ART / "error_analysis.csv", index=False)

    summary = {
        "data_audit": audit,
        "candidate_models": list(build_models().keys()),
        "selection_policy": "highest validation macro-F1; macro-recall tie-break",
        "selected_model": selected,
        "validation_results": val_df.to_dict(orient="records"),
        "escalation_policy": {
            "target_validation_auto_route_accuracy": 0.90,
            "threshold": float(threshold),
            "validation_auto_route_accuracy": float(auto_acc),
            "validation_auto_route_coverage": float(coverage),
            "confidence_definition": "max class probability for probabilistic models; top-vs-second decision margin gap for LinearSVC",
        },
        "test_result": {
            **test_metrics,
            "auto_route_accuracy": test_auto_accuracy,
            "auto_route_coverage": test_coverage,
            "human_escalation_rate": test_escalation_rate,
        },
        "claim_boundary": "offline intent classification and confidence-aware routing on Banking77; no guarantee of live support resolution, CSAT, cost reduction, or causal business lift",
    }
    (ART / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
