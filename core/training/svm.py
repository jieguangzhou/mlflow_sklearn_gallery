
from sklearn.svm import SVC
import mlflow

from core.metrics import eval_classification_metrics


def train_svc(train_x, train_y, test_x, test_y):
    mlflow.autolog()
    model = SVC()
    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return model, metrics
