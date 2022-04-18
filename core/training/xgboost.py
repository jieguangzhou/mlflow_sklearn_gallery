import mlflow
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from core.metrics import eval_classification_metrics
from core.utils import get_oridinal_encoder


def train_xgboost(train_x, train_y, test_x, test_y):
    pipeline_mods = []
    mlflow.autolog()
    pipeline_mods.append(('oridinal_encoder', get_oridinal_encoder()))

    model = XGBClassifier()
    pipeline_mods.append(('model', model))
    pipeline = Pipeline(steps=pipeline_mods)

    pipeline.fit(train_x, train_y)

    y_pred = pipeline.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return pipeline, metrics
