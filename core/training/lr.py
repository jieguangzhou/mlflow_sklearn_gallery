from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow

from core.metrics import eval_classification_metrics
from core.utils import get_onehot_encoder
from .params import LrParams


def train_lr(train_x, train_y, test_x, test_y, param_file=None, params=None):
    pipeline_mods = []
    mlflow.autolog()
    pipeline_mods.append(("onehot_encoder", get_onehot_encoder()))
    input_params = LrParams(
        LogisticRegression, param_file=param_file, param_str=params).input_params

    model = LogisticRegression(**input_params)
    pipeline_mods.append(("model", model))
    pipeline = Pipeline(steps=pipeline_mods)

    pipeline.fit(train_x, train_y)

    y_pred = pipeline.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return pipeline, metrics
