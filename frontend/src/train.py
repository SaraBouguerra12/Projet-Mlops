import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from preprocess import load_data


mlflow.set_experiment('vehicules_mlops')

X_train, X_test, y_train, y_test = load_data()

configs = [
    {
        'model_name': 'RandomForest_small',
        'type': 'rf',
        'n_estimators': 50,
        'max_depth': 3
    },
    {
        'model_name': 'RandomForest_large',
        'type': 'rf',
        'n_estimators': 200,
        'max_depth': 10
    },
    {
        'model_name': 'GradientBoosting',
        'type': 'gb',
        'n_estimators': 100,
        'learning_rate': 0.1
    }
]

for cfg in configs:

    with mlflow.start_run(run_name=cfg['model_name']):

        mlflow.log_params(cfg)

        if cfg['type'] == 'rf':
            model = RandomForestClassifier(
                n_estimators=cfg['n_estimators'],
                max_depth=cfg['max_depth']
            )
        elif cfg['type'] == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=cfg['n_estimators'],
                learning_rate=cfg['learning_rate']
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.sklearn.log_model(model, "model")

        print(f'{cfg["model_name"]} terminé | accuracy={accuracy:.4f}')