import pandas as pd
import numpy as np
import joblib, os, glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cols = [
    'COMPACTNESS','CIRCULARITY','DISTANCE_CIRCULARITY','RADIUS_RATIO',
    'PR_AXIS_ASPECT_RATIO','MAX_LENGTH_ASPECT_RATIO','SCATTER_RATIO',
    'ELONGATEDNESS','PR_AXIS_RECT','MAX_LENGTH_RECT',
    'SCALED_VARIANCE_MAJOR','SCALED_VARIANCE_MINOR',
    'SCALED_RADIUS_OF_GYRATION','SKEWNESS_MAJOR','SKEWNESS_MINOR',
    'KURTOSIS_MAJOR','KURTOSIS_MINOR','HOLLOWS_RATIO','CLASS'
]

# Charger les fichiers .dat
dfs = []
for f in sorted(glob.glob('data/raw/*.dat')):
    dfs.append(pd.read_csv(f, sep=r'\s+', header=None, names=cols))

if not dfs:
    print("ERREUR : aucun fichier .dat trouve dans data/raw/")
    print("Copie les fichiers xaa.dat ... xai.dat dans data/raw/")
    exit(1)

data = pd.concat(dfs, ignore_index=True)
data['CLASS'] = data['CLASS'].str.strip()
print("Dataset charge :", data.shape)

le = LabelEncoder()
y = le.fit_transform(data['CLASS'])
X = data.drop('CLASS', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(rf,     'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le,     'models/label_encoder.pkl')

acc = accuracy_score(y_test, rf.predict(X_test))
print("model.pkl         sauvegarde")
print("scaler.pkl        sauvegarde")
print("label_encoder.pkl sauvegarde")
print("Accuracy =", round(acc * 100, 2), "%")