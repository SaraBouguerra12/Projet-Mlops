from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(
    title="Vehicle Prediction API",
    version="1.0"
)

# --------------------------------------------------
# 🔥 MLflow configuration (IMPORTANT pour Docker)
# --------------------------------------------------
mlflow.set_tracking_uri("file:///app/mlruns")

MODEL_URI = "models:/vehicule_prediction_model/Production"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Vehicle Prediction API is running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(data: dict):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if "instances" not in data:
        raise HTTPException(status_code=400, detail="Missing 'instances' key")

    try:
        df = pd.DataFrame(data["instances"])

        # --------------------------------------------------
        # ✅ Vérification des features attendues
        # --------------------------------------------------
        expected_features = ["speed", "weight", "engine_size", "fuel"]

        missing = [col for col in expected_features if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing}"
            )

        # reorder columns
        df = df[expected_features]

        # --------------------------------------------------
        # prediction
        # --------------------------------------------------
        preds = model.predict(df)

        return {
            "predictions": preds.tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )