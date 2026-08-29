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
    assert metrics["selected_model"] in {
        "majority_baseline", "multinomial_nb", "logistic_regression", "linear_svc"
    }
    assert 0 <= metrics["test_result"]["macro_f1"] <= 1
    assert 0 <= metrics["test_result"]["auto_route_accuracy"] <= 1
    assert 0 <= metrics["test_result"]["auto_route_coverage"] <= 1
    assert 0 <= metrics["test_result"]["human_escalation_rate"] <= 1
    assert abs(metrics["test_result"]["auto_route_coverage"] + metrics["test_result"]["human_escalation_rate"] - 1) < 1e-9

    val = pd.read_csv(root / "artifacts" / "validation_metrics.csv")
    assert set(val["model"]) == {
        "majority_baseline", "multinomial_nb", "logistic_regression", "linear_svc"
    }
    assert (root / "artifacts" / "model.joblib").exists()
    assert (root / "artifacts" / "test_classification_report.csv").exists()
    assert (root / "artifacts" / "test_confusion_matrix.csv").exists()
    assert (root / "artifacts" / "error_analysis.csv").exists()
