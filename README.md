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
│       └── index.js
│
├── models/                     # Modeles sauvegardes
│   ├── model.pkl               # Random Forest entraine
│   ├── scaler.pkl              # StandardScaler
│   └── label_encoder.pkl       # LabelEncoder (bus/opel/saab/van)
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

## Taches Completees

| Tache | Description | Statut |
|-------|-------------|--------|
| Tache 1 | Nettoyage et preparation des donnees | DONE |
| Tache 2 | Interface Frontend (React Dashboard) | DONE |
| Tache 3 | Experimentation ML + MLflow (KNN, SVM, RF, LR) | DONE |
| **Tache 4** | **Interpretation et Analyse du Random Forest** | **DONE** |

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
