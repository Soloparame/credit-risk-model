from src.utils import evaluate_model

def test_metrics_output_keys():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    y_proba = [0.9, 0.1, 0.4, 0.7]
    metrics = evaluate_model(y_true, y_pred, y_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics

def test_metrics_values_are_floats():
    y_true = [1, 0, 1]
    y_pred = [1, 0, 0]
    y_proba = [0.8, 0.2, 0.3]
    metrics = evaluate_model(y_true, y_pred, y_proba)

    for value in metrics.values():
        assert isinstance(value, float)
