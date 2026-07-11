"""Build a Feast-ready parquet file: entity id, features, event_timestamp."""
import pandas as pd
from datetime import datetime, timedelta

def main():
    train_df = pd.read_csv("data/train.csv")
    eval_df = pd.read_csv("data/eval.csv")
    full_df = pd.concat([train_df, eval_df], ignore_index=True)

    full_df["flower_id"] = full_df.index.astype(int)

    base_time = datetime.now()
    full_df["event_timestamp"] = [
        base_time - timedelta(minutes=len(full_df) - i) for i in range(len(full_df))
    ]

    feast_cols = [
        "flower_id", "sepal_length", "sepal_width",
        "petal_length", "petal_width", "target", "event_timestamp"
    ]
    full_df[feast_cols].to_parquet("feature_repo/iris_features.parquet", index=False)
    print(f"Saved {len(full_df)} rows to feature_repo/iris_features.parquet")

if __name__ == "__main__":
    main()