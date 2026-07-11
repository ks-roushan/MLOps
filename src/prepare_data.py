"""Load IRIS, do a train/test split, save as CSV (v1 = 80/20 split, seed=42)."""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.3
RANDOM_STATE = 7

def main():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "target"
    ]

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["target"]
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/eval.csv", index=False)
    print(f"train: {train_df.shape}, eval: {test_df.shape}")

if __name__ == "__main__":
    main()