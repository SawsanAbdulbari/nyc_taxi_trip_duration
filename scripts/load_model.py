import os
import pandas as pd
import joblib
from feature_engineering import prepare_data
from evaluation import predict_eval, residual_plot

def load_and_test_model(data_path):
    """
    Loads the Ridge Regression model and tests it on the test dataset.

    Args:
        data_path (str): Path to the dataset CSV file.
    """
    model_path = os.path.join('D:\\ml_projects\\project-nyc-taxi-trip-duration', 'models', 'ridge_regression_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    
    # Load the saved model
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    # Load and prepare the dataset
    data = pd.read_csv(data_path)
    data_prepared = prepare_data(data)

    # Features to be used for prediction
    features = [col for col in data_prepared.columns if col != 'log_trip_duration']

    # Ensure the model is evaluated only if 'log_trip_duration' is available in the test data
    if 'log_trip_duration' in data_prepared.columns:
        # Predictions
        y_pred = model.predict(data_prepared[features])
        
        # Evaluate the model
        predict_eval(model, data_prepared[features], data_prepared['log_trip_duration'], 'Test Data')
        
        # Residual Plot
        residual_plot(data_prepared['log_trip_duration'], y_pred, 'Test Data Residuals')
    else:
        raise ValueError("The dataset must contain the 'log_trip_duration' column for evaluation.")

if __name__ == '__main__':
    # Path to the test dataset
    data_path = os.path.join('D:\\ml_projects\\project-nyc-taxi-trip-duration', 'data', 'split', 'test.csv')

    # Load and test the saved model on the test dataset
    load_and_test_model(data_path)
