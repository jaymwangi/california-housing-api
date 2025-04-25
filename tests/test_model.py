"""
California Housing Model Evaluation Script

This script provides tools for evaluating trained regression models on the California housing dataset.
It includes performance metrics calculation, error analysis, and visualization capabilities.

Key Features:
- Model loading with error handling
- Comprehensive metrics reporting (MAE, RMSE, R², custom accuracy)
- Diagnostic visualizations:
  - Actual vs Predicted prices
  - Residual analysis
  - Feature importance (for supported models)
- Percentage-based error reporting relative to mean house price

Author: Glita Jay
Version: 1.1
"""

# ============================== IMPORTS ==============================================
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ============================== MODEL LOADING ========================================

def load_model(model_path: str) -> object:
    """
    Load a trained machine learning model from disk
    
    Args:
        model_path (str): Path to the saved model file (.pkl)
        
    Returns:
        object: Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For general loading errors
        
    Example:
        >>> model = load_model("../models/california_housing.pkl")
    """
    # Validate model file existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Critical Error: Model file '{model_path}' not found")
        
    try:
        # Load model using joblib
        model = joblib.load(model_path)
        print("✅ Model loaded successfully from:", model_path)
        return model
    except Exception as e:
        raise Exception(f"❌ Model Loading Failed: {str(e)}")

# ============================== MODEL EVALUATION =====================================

def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Comprehensive model evaluation with metrics and visualizations
    
    Args:
        model (object): Trained regression model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True target values
        
    Returns:
        None: Outputs metrics and plots to console
        
    Analysis Includes:
    - Key regression metrics
    - Error percentages relative to mean price
    - Prediction visualization
    - Residual analysis
    - Feature importance (for tree-based models)
    """
    # ------------------------ Prediction Generation --------------------------------
    y_pred = model.predict(X_test)
    
    # ------------------------ Metrics Calculation ----------------------------------
    # Core regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Percentage-based metrics
    mean_price = np.mean(y_test)
    mae_percentage = (mae / mean_price) * 100
    rmse_percentage = (rmse / mean_price) * 100
    model_accuracy = (1 - (mae / mean_price)) * 100
    
    # ------------------------ Results Presentation ---------------------------------
    print("\n📊 Comprehensive Model Evaluation")
    print("--------------------------------")
    print(f"Mean Absolute Error (MAE):       {mae:.2f} ({mae_percentage:.2f}%)")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} ({rmse_percentage:.2f}%)")
    print(f"R² Score:                        {r2:.2f}")
    print(f"Model Accuracy:                  {model_accuracy:.2f}%")
    
    # ------------------------ Visualization Section --------------------------------
    # Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Home Prices\n(Ideal Line Shown in Red)", pad=20)
    plt.xlabel("Actual Prices ($100,000)", labelpad=15)
    plt.ylabel("Predicted Prices ($100,000)", labelpad=15)
    plt.grid(alpha=0.3)
    plt.show()
    
    # Residual Analysis Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title("Residual Analysis\n(Zero Error Line Shown in Red)", pad=20)
    plt.xlabel("Predicted Values ($100,000)", labelpad=15)
    plt.ylabel("Residuals ($100,000)", labelpad=15)
    plt.grid(alpha=0.3)
    plt.show()
    
    # -------------------- Feature Importance (Conditional) ------------------------
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title("Feature Importance Ranking", pad=20)
        plt.xlabel("Importance Score", labelpad=15)
        plt.ylabel("Feature", labelpad=15)
        plt.grid(axis='x', alpha=0.3)
        plt.show()
    else:
        print("\n⚠️  Feature importance not available for this model type")
        
    print("\n✅ Evaluation Complete!")

# ============================== MAIN EXECUTION =======================================

if __name__ == "__main__":
    """
    Main execution flow for model evaluation
    
    Steps:
    1. Model loading from specified path
    2. California housing data retrieval
    3. Data preparation with selected features
    4. Comprehensive model evaluation
    
    Note: Uses sklearn's California housing dataset as default test data
    """
    
    # ------------------------ Configuration ----------------------------------------
    MODEL_PATH = "C:/Users/jaymw/OneDrive/Documents/Jupyter Notebook/AI and DATA/Projects/USA_California_Housing_Price_Prediction/model/model_selected.pkl"  # Absolute path to model
    SELECTED_FEATURES = ["MedInc", "AveRooms", "AveOccup", "Latitude", "Longitude"]
    
    try:
        # ------------------------ Model Initialization -----------------------------
        print("\n🔍 Loading Model...")
        model = load_model(MODEL_PATH)
        
        # ------------------------ Data Preparation ---------------------------------
        print("\n📦 Loading Test Data...")
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        test_df = pd.DataFrame(data.data, columns=data.feature_names)
        test_df['target'] = data.target
        
        # Feature selection and target separation
        X_test = test_df[SELECTED_FEATURES]
        y_test = test_df['target']
        print(f"✅ Loaded {len(test_df)} samples with {len(SELECTED_FEATURES)} features")
        
        # ------------------------ Model Evaluation ---------------------------------
        print("\n🧪 Starting Model Evaluation...")
        evaluate_model(model, X_test, y_test)
        
    except Exception as e:
        print(f"\n❌ Critical Error: {str(e)}")
        print("💡 Debugging Tips:")
        print("- Verify model file path and permissions")
        print("- Check sklearn/package versions")
        print("- Validate input data structure")
