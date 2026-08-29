from pathlib import Path
import json
import pandas as pd

from src.pipeline import main


def test_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "artifacts" / "metrics.json").read_text())

    assert metrics["data_audit"]["classes"] == 77
    assert metrics["data_audit"]["test_rows"] > 0
    assert metrics["data_audit"]["train_val_exact_text_overlap"] == 0
    assert metrics["data_audit"]["train_test_exact_text_overlap"] == 0
    assert metrics["selected_model"] in {
        "majority_baseline", "multinomial_nb", "logistic_regression", "linear_svc"
    }
    assert metrics["escalation_policy"]["threshold_search"] == "all distinct validation confidence values"
    assert metrics["escalation_policy"]["validation_auto_route_accuracy"] >= 0.90

    test = metrics["test_result"]
    for key in ["macro_f1", "weighted_f1", "macro_precision", "macro_recall", "auto_route_accuracy", "auto_route_coverage", "human_escalation_rate"]:
        assert 0 <= test[key] <= 1
    assert abs(test["auto_route_coverage"] + test["human_escalation_rate"] - 1) < 1e-9
    assert test["auto_route_errors"] >= 0

    val = pd.read_csv(root / "artifacts" / "validation_metrics.csv")
    assert set(val["model"]) == {
        "majority_baseline", "multinomial_nb", "logistic_regression", "linear_svc"
    }
    assert (root / "artifacts" / "model.joblib").exists()
    assert (root / "artifacts" / "test_classification_report.csv").exists()
    assert (root / "artifacts" / "test_confusion_matrix.csv").exists()
    assert (root / "artifacts" / "error_analysis.csv").exists()
    assert (root / "artifacts" / "auto_route_errors.csv").exists()
