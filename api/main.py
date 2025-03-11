from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
try:
    model = joblib.load("../model/model.pkl")  # Use joblib instead of pickle
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define a root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the California Housing Price Prediction API"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: dict):
    try:
        features = data.get("features")
        if not isinstance(features, list):
            raise HTTPException(status_code=400, detail="Invalid input format. 'features' must be a list.")

        # Convert input to numpy array
        input_array = np.array(features).reshape(1, -1)

        # Ensure model has a predict method
        if not hasattr(model, "predict"):
            raise RuntimeError("Loaded model does not have a predict method.")

        # Make prediction
        prediction = model.predict(input_array)
        return {"predicted_price": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
