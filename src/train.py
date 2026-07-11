"""Train a simple classifier on data/train.csv, save model to models/model.joblib."""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"

def main():
    train_df = pd.read_csv("data/train.csv")
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.joblib")
    print("Model saved to models/model.joblib")

if __name__ == "__main__":
    main()