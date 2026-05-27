import pandas as pd
import numpy as np
import glob
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# ── ETAPE 1 : Charger les données ─────────────────────────────────────────
print("Chargement des donnees...")
cols = [
    'COMPACTNESS','CIRCULARITY','DISTANCE_CIRCULARITY','RADIUS_RATIO',
    'PR_AXIS_ASPECT_RATIO','MAX_LENGTH_ASPECT_RATIO','SCATTER_RATIO',
    'ELONGATEDNESS','PR_AXIS_RECT','MAX_LENGTH_RECT',
    'SCALED_VARIANCE_MAJOR','SCALED_VARIANCE_MINOR',
    'SCALED_RADIUS_OF_GYRATION','SKEWNESS_MAJOR','SKEWNESS_MINOR',
    'KURTOSIS_MAJOR','KURTOSIS_MINOR','HOLLOWS_RATIO','CLASS'
]
dfs = []
for f in sorted(glob.glob('data/raw/*.dat')):
    dfs.append(pd.read_csv(f, sep=r'\s+', header=None, names=cols))

data = pd.concat(dfs, ignore_index=True)
data['CLASS'] = data['CLASS'].str.strip()
print("Dataset :", data.shape)

# ── ETAPE 2 : Preprocessing ───────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(data['CLASS'])
X  = data.drop('CLASS', axis=1)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Train:", X_train.shape, "| Test:", X_test.shape)

# ── ETAPE 3 : Configurer MLflow ───────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Projet16_Vehicle_Classification")

def log_model(name, model, params):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        cv   = cross_val_score(model, X_train, y_train, cv=5).mean()
        train_acc = accuracy_score(y_train, model.predict(X_train))

        # Log paramètres
        mlflow.log_params(params)

        # Log métriques
        mlflow.log_metric("accuracy",       round(acc, 4))
        mlflow.log_metric("f1_score",       round(f1,  4))
        mlflow.log_metric("precision",      round(prec,4))
        mlflow.log_metric("recall",         round(rec, 4))
        mlflow.log_metric("cv_score",       round(cv,  4))
        mlflow.log_metric("train_accuracy", round(train_acc, 4))
        mlflow.log_metric("biais",          round(1 - train_acc, 4))
        mlflow.log_metric("variance",       round(train_acc - acc, 4))

        # Log modèle
        mlflow.sklearn.log_model(model, artifact_path=name)

        print(name + " => Accuracy=" + str(round(acc*100,2)) + "%  F1=" + str(round(f1*100,2)) + "%")

# ════════════════════════════════════════════════════════════════
# TACHE 3 — Experimentation des algorithmes
# ════════════════════════════════════════════════════════════════
print("\n--- TACHE 3 : Experimentation ---")

# KNN
log_model("KNN_k3",         KNeighborsClassifier(n_neighbors=3),                               {"n_neighbors":3,"metric":"euclidean"})
log_model("KNN_k5",         KNeighborsClassifier(n_neighbors=5),                               {"n_neighbors":5,"metric":"euclidean"})
log_model("KNN_k7",         KNeighborsClassifier(n_neighbors=7, weights="distance"),            {"n_neighbors":7,"metric":"manhattan","weights":"distance"})

# SVM
log_model("SVM_RBF_C1",     SVC(kernel="rbf",    C=1.0,  probability=True),                    {"kernel":"rbf",   "C":1.0,  "gamma":"scale"})
log_model("SVM_RBF_C10",    SVC(kernel="rbf",    C=10.0, probability=True),                    {"kernel":"rbf",   "C":10.0, "gamma":"scale"})
log_model("SVM_Linear",     SVC(kernel="linear", C=1.0,  probability=True),                    {"kernel":"linear","C":1.0})

# Random Forest
log_model("RF_100_d15",     RandomForestClassifier(n_estimators=100, max_depth=15,  random_state=42), {"n_estimators":100,"max_depth":15,  "criterion":"gini"})
log_model("RF_200_entropy",  RandomForestClassifier(n_estimators=200, max_depth=20,  criterion="entropy", random_state=42), {"n_estimators":200,"max_depth":20,"criterion":"entropy"})
log_model("RF_500_free",     RandomForestClassifier(n_estimators=500, max_depth=None,random_state=42), {"n_estimators":500,"max_depth":"None","criterion":"gini"})

# Logistic Regression
log_model("LR_C1",          LogisticRegression(C=1.0,  max_iter=1000, random_state=42),        {"C":1.0,  "solver":"lbfgs","max_iter":1000})
log_model("LR_C10",         LogisticRegression(C=10.0, max_iter=1000, random_state=42),        {"C":10.0, "solver":"lbfgs","max_iter":1000})

# Decision Tree
log_model("DT_d5",          DecisionTreeClassifier(max_depth=5,    random_state=42),            {"max_depth":5,   "criterion":"gini"})
log_model("DT_d15",         DecisionTreeClassifier(max_depth=15,   random_state=42),            {"max_depth":15,  "criterion":"gini"})
log_model("DT_libre",       DecisionTreeClassifier(max_depth=None, random_state=42),            {"max_depth":"None","criterion":"gini"})

# ════════════════════════════════════════════════════════════════
# TACHE 4 — Analyse Biais-Variance Random Forest
# ════════════════════════════════════════════════════════════════
print("\n--- TACHE 4 : Analyse Biais-Variance RF ---")

configs_t4 = [
    (10,   2,    "RF_T4_n10_d2"),
    (10,   5,    "RF_T4_n10_d5"),
    (50,   5,    "RF_T4_n50_d5"),
    (100,  8,    "RF_T4_n100_d8"),
    (100,  15,   "RF_T4_n100_d15_OPTIMAL"),
    (100,  None, "RF_T4_n100_dNone"),
    (200,  15,   "RF_T4_n200_d15"),
    (200,  None, "RF_T4_n200_dNone"),
    (500,  None, "RF_T4_n500_dNone"),
]

for n_est, max_d, run_name in configs_t4:
    log_model(
        run_name,
        RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42),
        {"n_estimators": n_est, "max_depth": str(max_d)}
    )

# ════════════════════════════════════════════════════════════════
# TACHE 4 — Stabilite random_state
# ════════════════════════════════════════════════════════════════
print("\n--- TACHE 4 : Stabilite random_state ---")

for rs in [0, 7, 21, 42, 77, 100, 123, 200, 314, 999]:
    log_model(
        "RF_stability_seed" + str(rs),
        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=rs),
        {"n_estimators":100, "max_depth":15, "random_state":rs, "experience":"stabilite"}
    )

print("\nTous les runs enregistres dans MLflow !")
print("Va sur : http://localhost:5001")