from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import os
import time
import requests
import io

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Hugging Face Hub model details
MODEL_REPO = "GlitaJay/california-housing-model"
MODEL_FILE = "model_selected.pkl"

# Environment variable to distinguish between local and production
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Define the local model path
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # Navigate up to the project root
    "model",  # Navigate into the "model" directory
    "model_selected.pkl"  # Specify the model file
)

# Global variable to store the model
model = None

# Function to load the model
def load_model():
    global model
    if model is not None:
        return model

    try:
        if ENVIRONMENT == "production":
            # Download the model from Hugging Face Hub in production
            url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
            logger.info(f"🔍 Downloading model from Hugging Face Hub: {url}")
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            model = joblib.load(io.BytesIO(response.content))
            logger.info(f"✅ Model loaded from Hugging Face Hub! Type: {type(model)}")
        else:
            # Load the model from the local file system in development
            if not os.path.exists(LOCAL_MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH}")
            logger.info(f"🔍 Loading model from local file system: {LOCAL_MODEL_PATH}")
            model = joblib.load(LOCAL_MODEL_PATH)
            logger.info(f"✅ Model loaded from local file system! Type: {type(model)}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

    return model

# Define request format
class HousingInput(BaseModel):
    MedInc: float
    AveRooms: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Middleware for Logging Requests, Responses & Execution Time
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()  # Start timing request
    logger.info(f"📥 Incoming request: {request.method} {request.url}")

    if request.method == "POST":
        body = await request.body()
        logger.info(f"📦 Request Body: {body.decode()}")

    response = await call_next(request)
    process_time = time.time() - start_time  # Calculate processing time
    logger.info(f"📤 Response {response.status_code} (Processed in {process_time:.4f} seconds)")
    return response

# Home Route
@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API!"}

# Model Info Endpoint
@app.get("/model-info")
def model_info():
    try:
        model = load_model()
        return {
            "model": "House Price Prediction Model",
            "version": "1.0",
            "status": "Loaded"
        }
    except Exception as e:
        return {"error": str(e)}

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Prediction Endpoint
@app.post("/predict")
def predict_price(data: HousingInput):
    try:
        model = load_model()
        df = pd.DataFrame([data.model_dump()])  # Use model_dump() instead of dict()
        logger.info(f"✅ Input received: {df}")

        prediction = model.predict(df)
        logger.info(f"✅ Prediction: {prediction[0]}")

        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return {"error": str(e)}