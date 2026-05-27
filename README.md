# Projet 16 — Classification des Types de Vehicules
## Machine Learning Avancee — ING4 DS-B | Sarra Bouguerra

---

## Structure du Projet

```
projet_vehicules/
├── backend/                    # API Flask (Python)
│   ├── app.py                  # Serveur Flask + routes Tache 4
│   ├── metadata.json           # Config modele et features
│   └── requirements.txt        # Dependances Python
│
├── frontend/                   # Interface React
│   ├── public/index.html
│   └── src/
│       ├── App.js              # Dashboard complet avec onglet Tache 4
│       ├── serve_api.py        # API FastAPI — Tache 5
│       ├── Dockerfile          # Conteneurisation — Tache 5
│       └── index.js
│
├── models/                     # Modeles sauvegardes
│   ├── model.pkl               # Random Forest entraine
│   ├── scaler.pkl              # StandardScaler
│   └── label_encoder.pkl       # LabelEncoder (bus/opel/saab/van)
│
├── saved_model/                # Modele exporte depuis MLflow Registry
│
├── notebooks/
│   └── Projet_Mlops_Tache4.ipynb   # Notebook Google Colab Tache 4
│
├── reports/
│   └── Rapport_Tache4_Sarra_Bouguerra.docx  # Compte rendu Tache 4
│
├── data/
│   ├── raw/                    # Fichiers .dat originaux (xaa..xai)
│   └── processed/              # Donnees nettoyees
│
└── mlruns/                     # Runs MLflow (generes automatiquement)
```

---

---
## Taches Completees
| Tache | Description | Statut |
|-------|-------------|--------|
| Tache 1 | Nettoyage et preparation des donnees | ✅ DONE |
| Tache 2 | Interface Frontend (React Dashboard) | ✅ DONE |
| Tache 3 | Experimentation ML + MLflow (KNN, SVM, RF, LR) | ✅ DONE |
| **Tache 4** | **Interpretation et Analyse du Random Forest** | ✅ **DONE** |
| **Tache 5** | **Pipeline MLOps — Tracking, Registry, API, Drift** | ✅ **DONE** |
---
## Lancer le Projet
### Backend Flask
```bash
cd backend
pip install -r requirements.txt
python app.py
# => http://localhost:5000
```
### Frontend React
```bash
cd frontend
npm install
npm start
# => http://localhost:3000
```
### API FastAPI (Tache 5)
```bash
cd frontend/src
uvicorn serve_api:app --host 0.0.0.0 --port 8000 --reload
# => http://localhost:8000
# => http://localhost:8000/docs  (Swagger UI)
```
### MLflow UI (Tache 5)
```bash
mlflow ui --host 0.0.0.0 --port 5000
# => http://localhost:5000
```
### Docker (Tache 5)
```bash
cd frontend/src
docker build -t vehicle-ml-api .
docker run -p 8000:8000 vehicle-ml-api
```
---
## API Routes
### Routes existantes (Taches 1-3)
| Methode | Route | Description |
|---------|-------|-------------|
| POST | /predict | Prediction d'un vehicule |
| GET  | /info | Info sur le modele |
| GET  | /example/<type> | Valeurs exemple (bus/van/saab/opel) |
### Nouvelles routes Tache 4
| Methode | Route | Description |
|---------|-------|-------------|
| GET | /tache4/feature-importance | Q1 — Importance des 18 features |
| GET | /tache4/stability | Q2 — Stabilite selon random_state |
| GET | /tache4/confusion-matrix | Q3 — Matrice de confusion + erreurs |
| GET | /tache4/bias-variance | Q4 — Tableau biais-variance (9 configs) |
| GET | /tache4/rf-vs-dt | Q5 — Comparaison RF vs Arbre de Decision |
| GET | /tache4/summary | Resume complet Tache 4 |
### API FastAPI Tache 5
| Methode | Route | Description |
|---------|-------|-------------|
| GET  | / | Health check — API operationnelle |
| POST | /predict | Prediction via modele MLflow Production |
---
## Tache 5 — Pipeline MLOps Complet
### Objectifs
- Tracking des experiences avec **MLflow Tracking**
- Comparaison visuelle via **MLflow UI**
- Gestion des versions avec **MLflow Model Registry**
- Deploiement via **API REST FastAPI**
- Detection de **Data Drift** avec Evidently AI + KS-Test
- **Re-entrainement automatique** si drift detecte (seuil > 30%)

### Stack Technique Tache 5
| Outil | Role |
|-------|------|
| MLflow >= 2.10 | Tracking, Registry, Serving |
| FastAPI | API REST de prediction |
| Evidently AI | Detection de data drift |
| Docker | Conteneurisation du serving |
| scikit-learn | Modeles ML (RF, GB) |
| SciPy | KS-Test statistique |

### Modeles entraines (Tache 5)
| Run | Modele | n_estimators | max_depth | Accuracy | Selectionne |
|-----|--------|:---:|:---:|:---:|:---:|
| Run 1 | RandomForest small | 50 | 3 | ~82% | Non |
| Run 2 | RandomForest large | 200 | 10 | ~89% | ✅ Oui |
| Run 3 | GradientBoosting | 100 | — | ~86% | Non |

### Detection de Drift
- Librairie : **Evidently AI** — rapport HTML complet
- Test statistique : **KS-Test** (p-value < 0.05 = drift detecte)
- Seuil alerte : **15%** de features driftees
- Seuil re-entrainement : **30%** de features driftees

### Exemple de requete /predict
```json
POST http://localhost:8000/predict
{
  "instances": [
    {
      "speed": 120,
      "weight": 1500,
      "engine_size": 1600,
      "fuel": 1
    }
  ]
}
```
---
## Resultats Tache 4
### Q1 — Top 3 Features Importantes
1. RADIUS_RATIO — 17.8%
2. ELONGATEDNESS — 15.2%  
3. SCATTER_RATIO — 12.8%
### Q2 — Stabilite
- std = 0.83% sur 10 random_states => modele ROBUSTE
### Q3 — Erreurs
- Confusion principale : saab ↔ opel (geometrie similaire)
- Taux d'erreur : 5.79% (11/190)
### Q4 — Configuration Optimale
| n_estimators | max_depth | Train Acc | Test Acc | Biais | Variance | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 2 | 68.42% | 65.26% | 0.316 | 0.032 | Underfitting |
| 100 | 15 | 94.21% | 82.63% | 0.058 | 0.116 | **OPTIMAL** |
| 500 | None | 100% | 83.68% | 0.000 | 0.163 | Overfitting |
### Q5 — RF vs Decision Tree
- RF : 82.63% vs DT : 79.47% => **RF +3.16%**
- CV RF : 84.83% vs CV DT : 78.35% => **RF +6.48%**
---
## Dataset
- **Source** : UCI Machine Learning Repository — Vehicle Silhouettes
- **Observations** : 946
- **Features** : 18 (numeriques)
- **Classes** : bus, opel, saab, van (4 classes equilibrees)
---

## Lancer le Projet

### Backend Flask
```bash
cd backend
pip install -r requirements.txt
python app.py
# => http://localhost:5000
```

### Frontend React
```bash
cd frontend
npm install
npm start
# => http://localhost:3000
```

---

## API Routes

### Routes existantes (Taches 1-3)
| Methode | Route | Description |
|---------|-------|-------------|
| POST | /predict | Prediction d'un vehicule |
| GET  | /info | Info sur le modele |
| GET  | /example/<type> | Valeurs exemple (bus/van/saab/opel) |

### Nouvelles routes Tache 4
| Methode | Route | Description |
|---------|-------|-------------|
| GET | /tache4/feature-importance | Q1 — Importance des 18 features |
| GET | /tache4/stability | Q2 — Stabilite selon random_state |
| GET | /tache4/confusion-matrix | Q3 — Matrice de confusion + erreurs |
| GET | /tache4/bias-variance | Q4 — Tableau biais-variance (9 configs) |
| GET | /tache4/rf-vs-dt | Q5 — Comparaison RF vs Arbre de Decision |
| GET | /tache4/summary | Resume complet Tache 4 |

---

## Resultats Tache 4

### Q1 — Top 3 Features Importantes
1. RADIUS_RATIO — 17.8%
2. ELONGATEDNESS — 15.2%  
3. SCATTER_RATIO — 12.8%

### Q2 — Stabilite
- std = 0.83% sur 10 random_states => modele ROBUSTE

### Q3 — Erreurs
- Confusion principale : saab ↔ opel (geometrie similaire)
- Taux d'erreur : 5.79% (11/190)

### Q4 — Configuration Optimale
| n_estimators | max_depth | Train Acc | Test Acc | Biais | Variance |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 2 | 68.42% | 65.26% | 0.316 | 0.032 | Underfitting |
| 100 | 15 | 94.21% | 82.63% | 0.058 | 0.116 | **OPTIMAL** |
| 500 | None | 100% | 83.68% | 0.000 | 0.163 | Overfitting |

### Q5 — RF vs Decision Tree
- RF : 82.63% vs DT : 79.47% => **RF +3.16%**
- CV RF : 84.83% vs CV DT : 78.35% => **RF +6.48%**

---

## Dataset
- **Source** : UCI Machine Learning Repository — Vehicle Silhouettes
- **Observations** : 946
- **Features** : 18 (numeriques)
- **Classes** : bus, opel, saab, van (4 classes equilibrees)
