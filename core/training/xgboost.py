from xgboost import XGBClassifier
import mlflow
from sklearn.pipeline import Pipeline

from core.metrics import eval_classification_metrics
from core.utils import get_oridinal_encoer


def train_xgboost(train_x, train_y, test_x, test_y):
    pipeline_mods = []
    mlflow.autolog()
    pipeline_mods.append(('oridinal_encoer', get_oridinal_encoer()))

    model = XGBClassifier()
    pipeline_mods.append(('model', model))
    pipeline = Pipeline(steps=pipeline_mods)

    pipeline.fit(train_x, train_y)

    y_pred = pipeline.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return pipeline, metrics
