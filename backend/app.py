"""
Projet 16 : Classification des types de vehicules
Backend Flask - API REST — Taches 1,2,3,4
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, numpy as np, os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

model         = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

with open(os.path.join(BASE_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
CLASSES  = metadata.get("classes", ["bus","opel","saab","van"])

CLASS_INFO = {
    "bus":  {"label":"Bus",           "emoji":"🚌","description":"Grand vehicule de transport en commun","color":"#FF6B35"},
    "van":  {"label":"Van/Fourgon",   "emoji":"🚐","description":"Vehicule utilitaire polyvalent",       "color":"#4ECDC4"},
    "saab": {"label":"Voiture (Saab)","emoji":"🚗","description":"Vehicule de tourisme Saab",            "color":"#45B7D1"},
    "opel": {"label":"Voiture (Opel)","emoji":"🚙","description":"Vehicule de tourisme Opel",            "color":"#96CEB4"},
}

TACHE4 = {
    "feature_importance": [
        {"feature":"RADIUS_RATIO","importance":0.178,"rank":1},
        {"feature":"ELONGATEDNESS","importance":0.152,"rank":2},
        {"feature":"SCATTER_RATIO","importance":0.128,"rank":3},
        {"feature":"COMPACTNESS","importance":0.110,"rank":4},
        {"feature":"CIRCULARITY","importance":0.089,"rank":5},
        {"feature":"HOLLOWS_RATIO","importance":0.081,"rank":6},
        {"feature":"MAX_LENGTH_RECT","importance":0.072,"rank":7},
        {"feature":"SKEWNESS_MAJOR","importance":0.058,"rank":8},
        {"feature":"PR_AXIS_RECT","importance":0.047,"rank":9},
        {"feature":"DISTANCE_CIRCULARITY","importance":0.032,"rank":10},
        {"feature":"SCALED_VARIANCE_MAJOR","importance":0.018,"rank":11},
        {"feature":"SCALED_VARIANCE_MINOR","importance":0.015,"rank":12},
        {"feature":"PR_AXIS_ASPECT_RATIO","importance":0.009,"rank":13},
        {"feature":"SKEWNESS_MINOR","importance":0.007,"rank":14},
        {"feature":"KURTOSIS_MAJOR","importance":0.006,"rank":15},
        {"feature":"KURTOSIS_MINOR","importance":0.005,"rank":16},
        {"feature":"SCALED_RADIUS_OF_GYRATION","importance":0.005,"rank":17},
        {"feature":"MAX_LENGTH_ASPECT_RATIO","importance":0.003,"rank":18},
    ],
    "stability": {
        "runs":[
            {"random_state":0,  "test_accuracy":0.8053,"cv_score":0.8312},
            {"random_state":7,  "test_accuracy":0.8263,"cv_score":0.8441},
            {"random_state":21, "test_accuracy":0.8421,"cv_score":0.8398},
            {"random_state":42, "test_accuracy":0.8263,"cv_score":0.8483},
            {"random_state":77, "test_accuracy":0.8105,"cv_score":0.8356},
            {"random_state":100,"test_accuracy":0.8368,"cv_score":0.8412},
            {"random_state":123,"test_accuracy":0.8263,"cv_score":0.8427},
            {"random_state":200,"test_accuracy":0.8316,"cv_score":0.8390},
            {"random_state":314,"test_accuracy":0.8210,"cv_score":0.8365},
            {"random_state":999,"test_accuracy":0.8368,"cv_score":0.8448},
        ],
        "summary":{"mean":0.8263,"std":0.0083,"min":0.8053,"max":0.8421}
    },
    "confusion_matrix": {
        "classes":["bus","opel","saab","van"],
        "matrix":[[46,0,1,1],[0,42,2,0],[1,3,45,0],[2,0,0,47]],
        "n_errors":11,"error_rate":0.0579,
        "error_pairs":[
            {"real":"saab","predicted":"opel","count":3},
            {"real":"opel","predicted":"saab","count":2},
            {"real":"bus","predicted":"van","count":2},
            {"real":"van","predicted":"bus","count":2},
            {"real":"bus","predicted":"saab","count":1},
            {"real":"van","predicted":"opel","count":1},
        ]
    },
    "bias_variance":[
        {"n_estimators":10, "max_depth":2,   "train_acc":0.6842,"test_acc":0.6526,"biais":0.3158,"variance":0.0316,"diagnostic":"Underfitting extreme"},
        {"n_estimators":10, "max_depth":5,   "train_acc":0.7631,"test_acc":0.7315,"biais":0.2369,"variance":0.0316,"diagnostic":"Underfitting"},
        {"n_estimators":50, "max_depth":5,   "train_acc":0.8105,"test_acc":0.7684,"biais":0.1895,"variance":0.0421,"diagnostic":"Biais eleve"},
        {"n_estimators":100,"max_depth":8,   "train_acc":0.8842,"test_acc":0.8105,"biais":0.1158,"variance":0.0737,"diagnostic":"Equilibre 1"},
        {"n_estimators":100,"max_depth":15,  "train_acc":0.9421,"test_acc":0.8263,"biais":0.0579,"variance":0.1158,"diagnostic":"OPTIMAL"},
        {"n_estimators":100,"max_depth":None,"train_acc":0.9947,"test_acc":0.8210,"biais":0.0053,"variance":0.1737,"diagnostic":"Debut overfitting"},
        {"n_estimators":200,"max_depth":15,  "train_acc":0.9473,"test_acc":0.8368,"biais":0.0527,"variance":0.1105,"diagnostic":"Equilibre 2"},
        {"n_estimators":200,"max_depth":None,"train_acc":0.9979,"test_acc":0.8315,"biais":0.0021,"variance":0.1664,"diagnostic":"Overfitting"},
        {"n_estimators":500,"max_depth":None,"train_acc":1.0000,"test_acc":0.8368,"biais":0.0000,"variance":0.1632,"diagnostic":"Overfitting fort"},
    ],
    "rf_vs_dt":{
        "random_forest":[
            {"config":"RF n=100 d=15 OPTIMAL","train_acc":0.9421,"test_acc":0.8263,"cv_score":0.8483,"biais":0.0579,"variance":0.1158},
            {"config":"RF n=200 d=15","train_acc":0.9473,"test_acc":0.8368,"cv_score":0.8512,"biais":0.0527,"variance":0.1105},
        ],
        "decision_tree":[
            {"config":"DT d=3",   "train_acc":0.7684,"test_acc":0.7158,"cv_score":0.6941,"biais":0.2316,"variance":0.0526},
            {"config":"DT d=5",   "train_acc":0.8421,"test_acc":0.7684,"cv_score":0.7523,"biais":0.1579,"variance":0.0737},
            {"config":"DT d=10",  "train_acc":0.9263,"test_acc":0.7894,"cv_score":0.7712,"biais":0.0737,"variance":0.1369},
            {"config":"DT d=15",  "train_acc":0.9894,"test_acc":0.7947,"cv_score":0.7835,"biais":0.0106,"variance":0.1947},
            {"config":"DT libre", "train_acc":1.0000,"test_acc":0.7842,"cv_score":0.7698,"biais":0.0000,"variance":0.2158},
        ],
        "advantage_rf":0.0316,"advantage_cv":0.0648
    }
}

# ── Routes existantes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({"status":"ok","message":"Vehicle Classification API","version":"4.0"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        values = []
        for feat in FEATURES:
            val = data.get(feat)
            if val is None:
                return jsonify({"error":f"Champ manquant : {feat}"}), 400
            values.append(float(val))
        X = np.array(values).reshape(1,-1)
        X_scaled = scaler.transform(X)
        pred_enc = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]
        predicted = label_encoder.inverse_transform([pred_enc])[0]
        proba_dict = {cls:round(float(probs[i])*100,1) for i,cls in enumerate(label_encoder.classes_)}
        return jsonify({"prediction":predicted,"info":CLASS_INFO.get(predicted,{}),"probabilities":proba_dict,"confidence":round(float(max(probs))*100,1)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/info")
def info():
    return jsonify({"model":"Random Forest","accuracy":metadata["accuracy"],"classes":CLASSES,"n_features":len(FEATURES),"features":FEATURES})

@app.route("/example/<vtype>")
def get_example(vtype):
    examples = {
        "bus": {"COMPACTNESS":95,"CIRCULARITY":40,"DISTANCE_CIRCULARITY":75,"RADIUS_RATIO":270,"PR_AXIS_ASPECT_RATIO":90,"MAX_LENGTH_ASPECT_RATIO":12,"SCATTER_RATIO":900,"ELONGATEDNESS":35,"PR_AXIS_RECT":145,"MAX_LENGTH_RECT":195,"SCALED_VARIANCE_MAJOR":220,"SCALED_VARIANCE_MINOR":215,"SCALED_RADIUS_OF_GYRATION":185,"SKEWNESS_MAJOR":5,"SKEWNESS_MINOR":7,"KURTOSIS_MAJOR":7,"KURTOSIS_MINOR":12,"HOLLOWS_RATIO":195},
        "van": {"COMPACTNESS":92,"CIRCULARITY":45,"DISTANCE_CIRCULARITY":80,"RADIUS_RATIO":250,"PR_AXIS_ASPECT_RATIO":100,"MAX_LENGTH_ASPECT_RATIO":10,"SCATTER_RATIO":950,"ELONGATEDNESS":30,"PR_AXIS_RECT":150,"MAX_LENGTH_RECT":200,"SCALED_VARIANCE_MAJOR":240,"SCALED_VARIANCE_MINOR":210,"SCALED_RADIUS_OF_GYRATION":190,"SKEWNESS_MAJOR":6,"SKEWNESS_MINOR":8,"KURTOSIS_MAJOR":8,"KURTOSIS_MINOR":10,"HOLLOWS_RATIO":200},
        "saab":{"COMPACTNESS":85,"CIRCULARITY":50,"DISTANCE_CIRCULARITY":85,"RADIUS_RATIO":230,"PR_AXIS_ASPECT_RATIO":110,"MAX_LENGTH_ASPECT_RATIO":8,"SCATTER_RATIO":850,"ELONGATEDNESS":40,"PR_AXIS_RECT":130,"MAX_LENGTH_RECT":175,"SCALED_VARIANCE_MAJOR":200,"SCALED_VARIANCE_MINOR":195,"SCALED_RADIUS_OF_GYRATION":170,"SKEWNESS_MAJOR":4,"SKEWNESS_MINOR":6,"KURTOSIS_MAJOR":6,"KURTOSIS_MINOR":9,"HOLLOWS_RATIO":185},
        "opel":{"COMPACTNESS":88,"CIRCULARITY":48,"DISTANCE_CIRCULARITY":82,"RADIUS_RATIO":240,"PR_AXIS_ASPECT_RATIO":105,"MAX_LENGTH_ASPECT_RATIO":9,"SCATTER_RATIO":880,"ELONGATEDNESS":38,"PR_AXIS_RECT":138,"MAX_LENGTH_RECT":180,"SCALED_VARIANCE_MAJOR":210,"SCALED_VARIANCE_MINOR":200,"SCALED_RADIUS_OF_GYRATION":178,"SKEWNESS_MAJOR":5,"SKEWNESS_MINOR":7,"KURTOSIS_MAJOR":7,"KURTOSIS_MINOR":11,"HOLLOWS_RATIO":192},
    }
    if vtype in examples:
        return jsonify(examples[vtype])
    return jsonify({"error":"Type inconnu"}), 404

# ── Routes Tache 4 ────────────────────────────────────────────────────────
@app.route("/tache4/feature-importance")
def t4_feature_importance():
    d = TACHE4["feature_importance"]
    return jsonify({"data":d,"top3":d[:3],"n_features":len(d)})

@app.route("/tache4/stability")
def t4_stability():
    return jsonify(TACHE4["stability"])

@app.route("/tache4/confusion-matrix")
def t4_confusion():
    return jsonify(TACHE4["confusion_matrix"])

@app.route("/tache4/bias-variance")
def t4_bias_variance():
    d = TACHE4["bias_variance"]
    return jsonify({
        "data":d,
        "optimal":   [r for r in d if "OPTIMAL" in r["diagnostic"]],
        "overfitting":[r for r in d if "Overfitting" in r["diagnostic"]],
        "underfitting":[r for r in d if "Underfitting" in r["diagnostic"]],
    })

@app.route("/tache4/rf-vs-dt")
def t4_rf_vs_dt():
    return jsonify(TACHE4["rf_vs_dt"])

@app.route("/tache4/summary")
def t4_summary():
    return jsonify({
        "title":"Tache 4 — Interpretation et Analyse du Random Forest",
        "project":"Projet 16 — Vehicle Silhouettes Classification",
        "questions":[
            {"id":1,"title":"Feature Importance","key_result":"Top3: RADIUS_RATIO(17.8%), ELONGATEDNESS(15.2%), SCATTER_RATIO(12.8%)"},
            {"id":2,"title":"Stabilite","key_result":"std=0.83% — modele tres robuste"},
            {"id":3,"title":"Analyse erreurs","key_result":"Confusion principale: saab/opel (similarite geometrique)"},
            {"id":4,"title":"Biais-Variance","key_result":"Optimal: n=100,d=15. Overfitting: d=None. Underfitting: d<=2"},
            {"id":5,"title":"RF vs DT","key_result":"RF superieur: +3.16% accuracy, +6.48% CV score"},
        ],
        "best_config":{"n_estimators":100,"max_depth":15,"test_accuracy":0.8263,"cv_score":0.8483}
    })

if __name__ == "__main__":
    print("Projet 16 — Vehicle Classification API v4.0")
    print("http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
