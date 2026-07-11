"""Evaluate models/model.joblib on data/eval.csv, print + save metrics.json."""
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"

def evaluate(model_path="models/model.joblib", eval_path="data/eval.csv"):
    model = joblib.load(model_path)
    eval_df = pd.read_csv(eval_path)
    X_eval = eval_df[FEATURES]
    y_eval = eval_df[TARGET]

    preds = model.predict(X_eval)

    metrics = {
        "accuracy": accuracy_score(y_eval, preds),
        "precision": precision_score(y_eval, preds, average="macro"),
        "recall": recall_score(y_eval, preds, average="macro"),
        "f1": f1_score(y_eval, preds, average="macro"),
    }
    return metrics

def main():
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()