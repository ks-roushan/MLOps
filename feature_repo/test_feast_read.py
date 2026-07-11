from feast import FeatureStore

def main():
    store = FeatureStore(repo_path=".")
    features = store.get_online_features(
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],
        entity_rows=[{"flower_id": 0}, {"flower_id": 1}],
    ).to_dict()
    print(features)

if __name__ == "__main__":
    main()