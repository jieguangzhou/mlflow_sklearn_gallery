import mlflow
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from core.metrics import eval_classification_metrics
from core.utils import get_oridinal_encoder
from .params import LightGBMParams


def train_lightgbm(train_x, train_y, test_x, test_y, param_file=None, params=None):
    pipeline_mods = []
    mlflow.autolog()
    pipeline_mods.append(("oridinal_encoder", get_oridinal_encoder()))

    input_params = LightGBMParams(
        LGBMClassifier, param_file=param_file, param_str=params).input_params

    model = LGBMClassifier(**input_params)
    pipeline_mods.append(("model", model))
    pipeline = Pipeline(steps=pipeline_mods)

    pipeline.fit(train_x, train_y)

    y_pred = pipeline.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return pipeline, metrics
