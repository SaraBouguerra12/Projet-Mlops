import os
import subprocess

import numpy as np
import pandas as pd
import mlflow

from scipy import stats
from sklearn.model_selection import train_test_split

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

# ---------------------------------------------------
# MLFLOW
# ---------------------------------------------------

mlflow.set_experiment("monitoring_drift")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

path = os.path.join(BASE_DIR, "data", "cleaned_data.csv")

df = pd.read_csv(path)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# SIMULATION DRIFT
# ---------------------------------------------------

X_prod = X_test.copy()

num_cols = X_prod.select_dtypes(include=np.number).columns

for col in num_cols[:2]:

    X_prod[col] = (
        X_prod[col] * 1.6
        + np.random.normal(0, 0.5, len(X_prod))
    )

# ---------------------------------------------------
# DOSSIERS
# ---------------------------------------------------

os.makedirs("reports", exist_ok=True)

# ---------------------------------------------------
# START RUN
# ---------------------------------------------------

with mlflow.start_run(run_name="drift_detection"):

    # ---------------------------------------------------
    # EVIDENTLY REPORT
    # ---------------------------------------------------

    report = Report(
        metrics=[
            DataDriftPreset(),
            DatasetDriftMetric()
        ]
    )

    report.run(
        reference_data=X_train,
        current_data=X_prod
    )

    # sauvegarde html
    report_path = "reports/drift_report.html"

    report.save_html(report_path)

    mlflow.log_artifact(report_path)

    # ---------------------------------------------------
    # DRIFT METRICS
    # ---------------------------------------------------

    result = report.as_dict()

    dataset_drift = result["metrics"][1]["result"]["dataset_drift"]

    drift_share = result["metrics"][1]["result"]["share_of_drifted_columns"]

    mlflow.log_metric(
        "dataset_drift",
        int(dataset_drift)
    )

    mlflow.log_metric(
        "drift_share",
        drift_share
    )

    print(f"\nDataset drift : {dataset_drift}")
    print(f"Drift share   : {drift_share:.2%}")

    # ---------------------------------------------------
    # KS TEST
    # ---------------------------------------------------

    results = []

    for col in X_train.columns:

        stat, pvalue = stats.ks_2samp(
            X_train[col],
            X_prod[col]
        )

        drifted = pvalue < 0.05

        results.append({
            "feature": col,
            "p_value": pvalue,
            "drifted": drifted
        })

        mlflow.log_metric(
            f"ks_{col}",
            pvalue
        )

    df_results = pd.DataFrame(results)

    csv_path = "ks_drift_results.csv"

    df_results.to_csv(csv_path, index=False)

    mlflow.log_artifact(csv_path)

    print("\nKS TEST RESULTS")
    print(df_results)

    # ---------------------------------------------------
    # AUTO RETRAIN
    # ---------------------------------------------------

    SEUIL_DRIFT = 0.30
    SEUIL_WARN = 0.15

    if drift_share > SEUIL_DRIFT:

        print("\nCRITIQUE : retraining automatique")

        subprocess.run(
            ["python", "train.py"],
            check=True
        )

        mlflow.log_metric(
            "retrain_triggered",
            1
        )

    elif drift_share > SEUIL_WARN:

        print("\nWARNING : drift détecté")

        mlflow.log_metric(
            "retrain_triggered",
            0
        )

    else:

        print("\nModèle stable")

        mlflow.log_metric(
            "retrain_triggered",
            0
        )
        # -----------------------------
# AUTO RETRAIN SI DRIFT
# -----------------------------

if df_results['drifted'].any():

    print("\nWARNING : drift détecté")
    print("Lancement du retraining automatique...\n")

    subprocess.run(["python", "train.py"])

    subprocess.run(["python", "register_best_model.py"])

    print("\nNouveau modèle entraîné et enregistré.")

else:

    print("\nAucun drift critique détecté.")