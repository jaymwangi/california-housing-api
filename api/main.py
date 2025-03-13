from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import pickle
import pandas as pd
import os

app = FastAPI()

# Download and Load Model from Hugging Face
REPO_ID = "GlitaJay/california-housing-model"
MODEL_FILENAME = "model.pkl"

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Download the model dynamically
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, cache_dir="model")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define request format
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict_price(data: HousingInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(df)
    
    return {"predicted_price": prediction[0]}

@app.get("/")
def home():
    return {"message": "California Housing Price Prediction API"}
