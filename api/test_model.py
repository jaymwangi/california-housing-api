import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Function to load the model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Error: The file '{model_path}' does not exist.")
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        raise Exception(f"❌ Error loading the model: {e}")

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Convert errors to percentage relative to mean house price
    mean_price = np.mean(y_test)
    mae_percentage = (mae / mean_price) * 100
    rmse_percentage = (rmse / mean_price) * 100

    # Calculate model accuracy as percentage
    model_accuracy = (1 - (mae / mean_price)) * 100
    error_percentage = 100 - model_accuracy

    # Print evaluation results
    print("📊 Model Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} ({mae_percentage:.2f}%)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} ({rmse_percentage:.2f}%)")
    print(f"R² Score: {r2:.2f}")
    print(f"Model Accuracy: {model_accuracy:.2f}%")
    print(f"Error Percentage: {error_percentage:.2f}%")

    # Visualize the predictions vs actual values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.show()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    # Feature importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
        plt.title("Feature Importance")
        plt.show()
    else:
        print("⚠️ Feature importance not available for this model.")

    print("✅ Testing complete!")

# Main script
if __name__ == "__main__":
    # Path to the model file (relative to the script location)
    model_path = "../model/model_selected.pkl"  # Updated path to the model file

    try:
        # Load the model
        model = load_model(model_path)

        # Load the test dataset
        from sklearn.datasets import fetch_california_housing

        # Fetch the dataset
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        # Define the selected features
        selected_features = ["MedInc", "AveRooms", "AveOccup", "Latitude", "Longitude"]

        # Prepare the test data
        X_test = df[selected_features]
        y_test = df['target']

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

    except Exception as e:
        print(e)