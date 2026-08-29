from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.data import load_dataset
from src.pipeline import confidence_score


ROOT = Path(__file__).resolve().parents[1]


def test_premerge_hardening_invariants():
    """Strict independent checks over data, saved model, metrics, and routing policy."""
    train, val, test, audit, classes = load_dataset(42)

    # Dataset integrity and split invariants.
    assert len(train) == 8002
    assert len(val) == 2001
    assert len(test) == 3080
    assert len(classes) == 77
    assert set(train["category"]) == set(classes)
    assert set(val["category"]) == set(classes)
    assert set(test["category"]) == set(classes)
    assert train["text"].str.strip().ne("").all()
    assert val["text"].str.strip().ne("").all()
    assert test["text"].str.strip().ne("").all()
    assert audit["train_val_exact_text_overlap"] == 0
    assert audit["train_test_exact_text_overlap"] == 0

    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    artifact = joblib.load(ROOT / "artifacts" / "model.joblib")
    model = artifact["model"]
    threshold = float(artifact["escalation_threshold"])

    # Selection must materially beat the trivial baseline and avoid a suspicious
    # validation/test collapse.
    rows = {row["model"]: row for row in metrics["validation_results"]}
    assert metrics["selected_model"] == "linear_svc"
    assert rows["linear_svc"]["macro_f1"] > rows["majority_baseline"]["macro_f1"] + 0.80
    assert rows["linear_svc"]["macro_f1"] >= rows["logistic_regression"]["macro_f1"]
    assert abs(metrics["test_result"]["macro_f1"] - rows["linear_svc"]["macro_f1"]) < 0.05

    # Recompute validation routing independently from the serialized model.
    val_pred = model.predict(val["text"])
    val_conf = confidence_score(model, val["text"])
    assert np.isfinite(val_conf).all()
    assert (val_conf >= 0).all()
    val_auto = val_conf >= threshold
    assert val_auto.any()
    val_acc = float(np.mean(val["category"].to_numpy()[val_auto] == val_pred[val_auto]))
    val_cov = float(val_auto.mean())
    assert val_acc >= 0.90
    assert abs(val_acc - metrics["escalation_policy"]["validation_auto_route_accuracy"]) < 1e-12
    assert abs(val_cov - metrics["escalation_policy"]["validation_auto_route_coverage"]) < 1e-12

    # Apply the locked validation threshold unchanged to the official test set.
    test_pred = model.predict(test["text"])
    test_conf = confidence_score(model, test["text"])
    assert np.isfinite(test_conf).all()
    assert (test_conf >= 0).all()
    test_auto = test_conf >= threshold
    test_acc = float(np.mean(test["category"].to_numpy()[test_auto] == test_pred[test_auto]))
    test_cov = float(test_auto.mean())
    test_errors = int((test_auto & (test["category"].to_numpy() != test_pred)).sum())
    assert test_acc >= 0.90
    assert test_cov >= 0.90
    assert abs(test_acc - metrics["test_result"]["auto_route_accuracy"]) < 1e-12
    assert abs(test_cov - metrics["test_result"]["auto_route_coverage"]) < 1e-12
    assert test_errors == metrics["test_result"]["auto_route_errors"]

    # Artifact outputs must be complete and internally consistent.
    cm = pd.read_csv(ROOT / "artifacts" / "test_confusion_matrix.csv", index_col=0)
    report = pd.read_csv(ROOT / "artifacts" / "test_classification_report.csv", index_col=0)
    auto_errors = pd.read_csv(ROOT / "artifacts" / "auto_route_errors.csv")
    assert cm.shape == (77, 77)
    assert int(cm.to_numpy().sum()) == 3080
    assert set(classes).issubset(set(report.index))
    assert len(auto_errors) <= 100
    if len(auto_errors):
        assert auto_errors["auto_route"].astype(bool).all()
        assert (~auto_errors["correct"].astype(bool)).all()
        assert (auto_errors["confidence"] >= threshold).all()
