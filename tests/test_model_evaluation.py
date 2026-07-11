"""Model evaluation tests: load model, run inference, assert metric thresholds."""
import os
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "models/model.joblib"
EVAL_PATH = "data/eval.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"

# Minimum acceptable thresholds -- CI fails the build if the model dips below these
MIN_ACCURACY = 0.85
MIN_PRECISION = 0.85
MIN_RECALL = 0.85
MIN_F1 = 0.85


@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH} -- did dvc pull run?"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def eval_data():
    assert os.path.exists(EVAL_PATH), f"Eval data not found at {EVAL_PATH} -- did dvc pull run?"
    df = pd.read_csv(EVAL_PATH)
    return df[FEATURES], df[TARGET]


@pytest.fixture(scope="module")
def predictions(model, eval_data):
    X_eval, _ = eval_data
    return model.predict(X_eval)


def test_model_loads(model):
    assert model is not None


def test_predictions_shape(predictions, eval_data):
    _, y_eval = eval_data
    assert len(predictions) == len(y_eval), "Prediction count doesn't match eval set size"


def test_predictions_valid_classes(predictions):
    valid_classes = {0, 1, 2}
    predicted_classes = set(predictions)
    assert predicted_classes.issubset(valid_classes), (
        f"Model predicted invalid classes: {predicted_classes - valid_classes}"
    )


def test_accuracy_above_threshold(predictions, eval_data):
    _, y_eval = eval_data
    acc = accuracy_score(y_eval, predictions)
    assert acc >= MIN_ACCURACY, f"Accuracy {acc:.4f} below threshold {MIN_ACCURACY}"


def test_precision_above_threshold(predictions, eval_data):
    _, y_eval = eval_data
    prec = precision_score(y_eval, predictions, average="macro", zero_division=0)
    assert prec >= MIN_PRECISION, f"Precision {prec:.4f} below threshold {MIN_PRECISION}"


def test_recall_above_threshold(predictions, eval_data):
    _, y_eval = eval_data
    rec = recall_score(y_eval, predictions, average="macro", zero_division=0)
    assert rec >= MIN_RECALL, f"Recall {rec:.4f} below threshold {MIN_RECALL}"


def test_f1_above_threshold(predictions, eval_data):
    _, y_eval = eval_data
    f1 = f1_score(y_eval, predictions, average="macro", zero_division=0)
    assert f1 >= MIN_F1, f"F1 {f1:.4f} below threshold {MIN_F1}"


def test_no_class_ignored(predictions, eval_data):
    """Sanity check: model shouldn't collapse to predicting only 1-2 classes."""
    _, y_eval = eval_data
    n_true_classes = y_eval.nunique()
    n_pred_classes = pd.Series(predictions).nunique()
    assert n_pred_classes >= n_true_classes - 1, (
        f"Model only predicted {n_pred_classes} classes; expected close to {n_true_classes}"
    )