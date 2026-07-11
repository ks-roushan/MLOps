from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

flower = Entity(name="flower_id", join_keys=["flower_id"])

iris_source = FileSource(
    path="iris_features.parquet",
    timestamp_field="event_timestamp",
)

iris_feature_view = FeatureView(
    name="iris_features",
    entities=[flower],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=iris_source,
    online=True,
)