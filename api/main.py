"""
California Housing Price Prediction API with Hugging Face Hub Integration

This FastAPI application provides a RESTful interface for predicting California housing prices using
a machine learning model loaded either from Hugging Face Hub (in production) or local storage (in development).

Architecture Highlights:
- Modular design with clear separation of concerns
- Environment-aware configuration (development/production)
- Async/await pattern for non-blocking I/O operations
- State management using FastAPI's app.state pattern
- Comprehensive input validation with Pydantic
- Production-grade logging and monitoring

Features:
- Environment-specific model loading (Hugging Face Hub or local file)
- Thread-safe model initialization using asyncio locks
- Request/response lifecycle logging
- Pydantic validation with business logic checks
- Health checks and model status monitoring
- Error handling with contextual HTTP status codes
- Type hints throughout for better maintainability

Author: Glita Jay
Version: 1.1
Date: [Insert Date]
"""

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
import joblib
import pandas as pd
import logging
import os
import time
import asyncio
import sklearn
import torch
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager

# ============================== 🚀 App Initialization ===============================
# Initialize FastAPI application with OpenAPI metadata for documentation
app = FastAPI(
    title="California Housing Price Prediction API",
    description="API for predicting median house values in California districts",
    version="1.1",
    # OpenAPI documentation customization
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================== 🧾 Logging Setup ====================================
# Configure structured logging for observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================== ⚙️ Configuration =====================================
# Environment variables and model configuration
# (All sensitive configuration should come from environment variables in production)
MODEL_REPO = "GlitaJay/california-housing-model"  # Hugging Face Model Repository ID
MODEL_FILE = "model_selected.pkl"                 # Serialized model filename
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # Runtime environment detection

# Local model path configuration (for development/testing)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "..", "model", MODEL_FILE)

# ============================== 📦 Global Model State ================================
# Model state management using FastAPI's app.state pattern
# - model: Loaded scikit-learn model instance
# - model_loaded: Boolean flag indicating load status
# - model_lock: Async lock for thread-safe initialization
model = None
model_loaded = False
model_lock = asyncio.Lock()  # Async lock for concurrent access control

# ============================== 🔍 Model Loader ======================================
async def load_model(app: FastAPI) -> sklearn.base.BaseEstimator:
    """
    Asynchronous model loader with environment-aware initialization
    
    Implements:
    - Production: Downloads from Hugging Face Hub using authenticated API
    - Development: Loads from local file system
    - Thread-safe initialization using async locks
    - Error handling and status reporting
    
    Args:
        app (FastAPI): FastAPI application instance for state management
    
    Returns:
        sklearn.base.BaseEstimator: Loaded scikit-learn compatible model
    
    Raises:
        HTTPException: 500 error with details for any loading failure
        FileNotFoundError: If local model file is missing in development mode
    """
    async with model_lock:  # Ensure single-threaded model initialization
        # Return existing model if already loaded
        if hasattr(app.state, 'model') and app.state.model is not None:
            return app.state.model

        try:
            temp_model = None
            logger.info(f"🏗️  Initializing model in {ENVIRONMENT} environment")

            if ENVIRONMENT == "production":
                # Production: Lazy load model on demand (when required)
                if not hasattr(app.state, 'model') or app.state.model is None:
                    logger.info("🚀 Initializing production model from Hugging Face Hub")
                    model_path = hf_hub_download(
                        repo_id=MODEL_REPO,
                        filename=MODEL_FILE,
                        cache_dir="model_cache",  # Local cache directory
                        resume_download=True      # Resume interrupted downloads
                    )
                    logger.info(f"✅ Model downloaded to: {model_path}")
                    temp_model = joblib.load(model_path)
                    logger.info("👍 Hugging Face model loaded successfully")
                    app.state.model = temp_model
                    app.state.model_loaded = True

            else:
                # Development: Load from local file system
                if not os.path.exists(LOCAL_MODEL_PATH):
                    raise FileNotFoundError(
                        f"❌ Local model not found at {LOCAL_MODEL_PATH}. "
                        "Run model training or check file paths."
                    )
                
                logger.info(f"🔄 Loading local model from {LOCAL_MODEL_PATH}")
                temp_model = joblib.load(LOCAL_MODEL_PATH)
                logger.info("✅ Local model loaded successfully")
                app.state.model = temp_model
                app.state.model_loaded = True

            return app.state.model

        except Exception as e:
            error_msg = f"❌ Model loading failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            app.state.model_loaded = False
            raise HTTPException(
                status_code=500,
                detail=error_msg,
                headers={"X-Error-Message": error_msg}
            )


# ============================== 💡 Lifespan Context Manager =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for resource initialization/cleanup
    
    Responsibilities:
    - Model loading during startup
    - State management initialization
    - Graceful shutdown handling
    
    Flow:
    1. On startup: Load model and set application state
    2. Yield control to application
    3. On shutdown: Cleanup resources (currently no explicit cleanup needed)
    
    Safety Features:
    - Handles model loading failures gracefully
    - Sets model_loaded flag appropriately
    - Implements comprehensive logging
    """
    logger.info("🟢 Application startup initiated")
    startup_success = False

    try:
        # Model initialization sequence
        app.state.model = await load_model(app)
        app.state.model_loaded = True
        startup_success = True
        logger.info("✅ Model successfully loaded and set in app state")
    except Exception as e:
        logger.critical(f"❌ Failed to load model during startup: {e}")
        app.state.model_loaded = False
        startup_success = False
    finally:
        yield  # Application runs here
        # Cleanup logic would go here if needed
        logger.info("🔴 Application shutdown completed")

# Reinitialize app with lifespan management
app = FastAPI(lifespan=lifespan)

# ============================== 📥 Input Schema ======================================

class HousingFeatures(BaseModel):
    """
    Input model for California housing price prediction.

    Fields:
        MedInc: Median income in block group (log scaled, 10k USD units).
        AveRooms: Average rooms per household (capped at 50).
        AveOccup: Average occupants per household (range: 0.5 - 10).
        Latitude: Latitude within California (32.5°N to 42.0°N).
        Longitude: Longitude within California (-124.5°W to -113.5°W).
    """

    MedInc: float = Field(..., gt=0, example=3.8479, description="Median income in block group (log scaled, 10k USD units)")
    AveRooms: float = Field(..., gt=0, example=5.069, description="Average rooms per household (capped at 50)")
    AveOccup: float = Field(..., gt=0, example=2.742, description="Average occupants per household (0.5-10 range)")
    Latitude: float = Field(..., ge=32.5, le=42.0, example=34.42, description="WGS84 latitude (California boundaries)")
    Longitude: float = Field(..., ge=-124.5, le=-113.5, example=-118.32, description="WGS84 longitude (California boundaries)")

    @staticmethod
    def _check_valid_numeric(value, field_name: str) -> float:
        """
        Validates that a value is numeric and not empty or malformed.

        Args:
            value: The input value to validate.
            field_name: The name of the field being checked (for error reporting).

        Returns:
            A valid float value.

        Raises:
            ValueError: If the input is null, empty, or not convertible to float.
        """
        if value is None:
            raise ValueError(f"{field_name} cannot be null")
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty or just whitespace")
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"{field_name} must be a valid number")
        return float(value)

    @field_validator("Latitude")
    def check_latitude_in_range(cls, v):
        """Ensures latitude is numeric and falls within California's valid range."""
        v = cls._check_valid_numeric(v, "Latitude")
        if not (32.5 <= v <= 42.0):
            raise ValueError("Latitude must be between 32.5°N and 42.0°N")
        return v

    @field_validator("Longitude")
    def check_longitude_in_range(cls, v):
        """Ensures longitude is numeric and falls within California's valid range."""
        v = cls._check_valid_numeric(v, "Longitude")
        if not (-124.5 <= v <= -113.5):
            raise ValueError("Longitude must be between -124.5°W and -113.5°W")
        return v

    @field_validator("MedInc")
    def validate_income(cls, v):
        """Validates that income is positive, numeric, and below a reasonable upper limit."""
        v = cls._check_valid_numeric(v, "MedInc")
        if v > 25:
            raise ValueError("MedInc exceeds realistic threshold for California")
        return v

    @field_validator("AveRooms")
    def validate_rooms(cls, v):
        """Validates room count is numeric, non-empty, and <= 50."""
        v = cls._check_valid_numeric(v, "AveRooms")
        if v > 50:
            raise ValueError("AveRooms must not exceed 50")
        return v

    @field_validator("AveOccup")
    def validate_occupancy(cls, v):
        """Validates occupancy is numeric and within a realistic household range (0.5 to 10)."""
        v = cls._check_valid_numeric(v, "AveOccup")
        if not (0.5 <= v <= 10):
            raise ValueError("AveOccup must be in the range 0.5 - 10")
        return v

    @model_validator(mode='before')
    def logical_consistency_checks(cls, values):
        """
        Performs cross-field logic checks for consistency:
        - High rooms with very low income is flagged as suspicious.
        - Very low income near the coast triggers a warning.

        Returns:
            The validated values dict.

        Raises:
            ValueError: If essential fields are missing or logic rules are violated.
        """
        income = values.get("MedInc")
        rooms = values.get("AveRooms")
        lat = values.get("Latitude")
        lon = values.get("Longitude")

        if any(v is None for v in [income, rooms, lat, lon]):
            raise ValueError("Missing critical values for logical consistency check")

        if rooms > 10 and income < 2:
            raise ValueError("Suspicious input: Many rooms but low income")

        if lon > -118.5 and income < 1.5:
            logger.warning("⚠️ Low income for coastal region - verify input")

        return values

    class Config:
        extra = "forbid"
        validate_assignment = True


# ============================== 📊 Request Middleware ================================
@app.middleware("http")
async def request_auditor(request: Request, call_next):
    """
    Request auditing middleware that:
    - Logs incoming requests with unique IDs
    - Tracks processing time
    - Captures request bodies for debugging
    - Logs response status and performance
    
    Security Note: 
    - Body logging is debug-only
    - Sensitive data should be filtered in production
    """
    request_id = f"{time.time()}-{hash(request)}"
    start_time = time.time()
    
    # Request logging
    logger.info(f"📥 Incoming {request.method} {request.url.path} | ID: {request_id}")
    
    if request.method == "POST" and logger.isEnabledFor(logging.DEBUG):
        body = await request.body()
        logger.debug(f"📜 Request body: {body.decode()}")
    
    # Process request
    response = await call_next(request)
    
    # Response logging
    duration = time.time() - start_time
    logger.info(
        f"📤 Response {request_id} | "
        f"Status: {response.status_code} | "
        f"Duration: {duration:.2f}s"
    )
    
    # Add performance header
    response.headers["X-Processing-Time"] = f"{duration:.4f}s"
    return response

# ============================== 🔍 Monitoring Endpoints ==============================
@app.get("/", tags=["Monitoring"], summary="Service health check")
async def service_root():
    """Root endpoint for quick service status verification"""
    return {
        "service": "California Housing Price Prediction",
        "version": app.version,
        "status": "operational",
        "environment": ENVIRONMENT,
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/model/status", tags=["Model"], summary="Model metadata endpoint")
async def model_status_check():
    """
    Provides detailed model status information including:
    - Load status
    - Model type
    - Environment
    - Source location
    """
    if not app.state.model_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": "Model not loaded"}
        )
    
    # Assuming model has an attribute 'model_path' that stores the file path
    model_file_path = app.state.model.model_path if hasattr(app.state.model, 'model_path') else None

    if model_file_path:
        model_size = f"{os.path.getsize(model_file_path) / 1024:.2f} KB"
    else:
        model_size = "Unknown"

    return {
        "status": "loaded",
        "model_type": str(type(app.state.model)),
        "environment": ENVIRONMENT,
        "model_source": "Hugging Face Hub" if ENVIRONMENT == "production" else "Local",
        "model_size": model_size
    }


@app.get("/health", tags=["Monitoring"], summary="System health check")
async def health_check():
    """
    Comprehensive health check endpoint verifying:
    - Model availability
    - Memory utilization
    - Environment status
    """
    model_status = "loaded" if app.state.model_loaded else "unavailable"
    model_memory = "n/a"
    
    if app.state.model_loaded:
        # For PyTorch models, you can use torch.cuda.memory_allocated() if using GPU
        if isinstance(app.state.model, torch.nn.Module):
            if torch.cuda.is_available():
                model_memory = f"{torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB"
            else:
                model_memory = f"{os.sys.getsizeof(app.state.model) / 1024 / 1024:.2f}MB"
        # If model is not PyTorch, try default model size calculations
        else:
            model_size = os.sys.getsizeof(app.state.model)
            model_memory = f"{model_size / 1024 / 1024:.2f}MB"

    return {
        "status": "healthy" if app.state.model_loaded else "degraded",
        "components": {
            "model": {
                "status": model_status,
                "memory": model_memory
            },
            "environment": ENVIRONMENT,
            "system_memory": f"{os.sys.getsizeof([]) / 1024 / 1024:.2f}MB used"
        }
    }


# ============================== 🧠 Prediction Endpoint ==============================
@app.post("/predict", tags=["Prediction"], summary="House price prediction")
async def predict_house_price(features: HousingFeatures, request: Request):
    """
    Main prediction endpoint that:
    - Validates input using HousingFeatures model
    - Converts input to model-compatible format
    - Makes prediction using loaded model
    - Returns standardized response
    
    Response Includes:
    - Predicted value (scaled)
    - Input features echo
    - Model version
    - Prediction units
    
    Error Handling:
    - 503 if model not loaded
    - 422 for invalid input
    - 500 for prediction errors
    """
    # Model availability check
    if not app.state.model_loaded:
        logger.error("🚨 Prediction attempt with unloaded model")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "detail": "Model not initialized - try again later"
            }
        )

    try:
        # Input processing
        input_data = features.dict()
        logger.debug(f"🔍 Prediction input: {input_data}")
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([input_data], columns=[
            "MedInc", "AveRooms", "AveOccup", "Latitude", "Longitude"
        ])
        
        # Model prediction
        model = app.state.model
        logger.info("🏃 Starting prediction...")
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)  # Round to nearest $10k
        
        logger.info(f"📈 Prediction result: ${prediction * 100000:.2f}")
        
        return {
            "prediction": prediction,
            "units": "hundred-thousands of USD (scaled)",
            "input": input_data,
            "model_version": app.version,
            "environment": ENVIRONMENT
        }

    except Exception as e:
        logger.exception("🔥 Prediction error stack trace")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
            headers={"X-Error": "PredictionError"}
        )