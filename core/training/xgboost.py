from lightgbm import LGBMClassifier
import mlflow

from core.metrics import eval_classification_metrics


def train_xgboost(train_x, train_y, test_x, test_y):
    mlflow.autolog()
    model = LGBMClassifier()
    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return model, metrics
