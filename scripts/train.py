import pandas as pd
import numpy as np
import os
import joblib
from utils import extract_datetime_features, filter_geographic_bounds, distance, manhattan_distance, bearing_array, create_airport_features, remove_outliers
from feature_engineering import prepare_data
from model import build_pipeline, train_model
from evaluation import predict_eval, cross_val_eval, residual_plot, distribution_comparison

# Set the root directory for data files
root_dir = 'D:\\ml_projects\\project-nyc-taxi-trip-duration\\data\\split'

# Load training and validation data
train = pd.read_csv(os.path.join(root_dir, 'train.csv'))
test = pd.read_csv(os.path.join(root_dir, 'val.csv'))

# Prepare the training data
train = prepare_data(train)

# Define feature sets
numeric_features = ['pickup_latitude', 
                    'pickup_longitude',
                    'dropoff_latitude', 
                    'dropoff_longitude',
                    'distance_haversine', 
                    'distance_manhattan', 
                    'direction',
                    'trip_speed']

categorical_features = ['dayofweek', 
                        'month', 
                        'hour', 
                        'day', 
                        'vendor_id',
                        'passenger_count', 
                        'store_and_fwd_flag',
                        'pickup_jfk', 
                        'pickup_lga', 
                        'pickup_ewr',
                        'dropoff_jfk', 
                        'dropoff_lga', 
                        'dropoff_ewr']

train_features = numeric_features + categorical_features

missing_features = [feat for feat in train_features if feat not in train.columns]
if missing_features:
    raise ValueError(f"Missing features in the train set: {missing_features}")

# Separate features and target
X_train = train[train_features]
y_train = train['log_trip_duration']

# Build the pipeline
pipeline = build_pipeline(categorical_features, numeric_features)

# Train the model
model = train_model(pipeline, X_train, y_train)

# Evaluate the model using cross-validation
cv_rmse, cv_r2 = cross_val_eval(pipeline, X_train, y_train)

# Print cross-validation results
print(f"Cross-Validation Evaluation:\n  Mean RMSE: {cv_rmse:.4f}\n  Mean R²: {cv_r2:.4f}")

# Prepare the test data
test = prepare_data(test)
X_test = test[train_features]
y_test = test['log_trip_duration']

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model on the train data
predict_eval(model, X_train, y_train, 'Train Set')

# Evaluate the model on the test data
predict_eval(model, X_test, y_test, 'Test Set')

# Residual Plots
residual_plot(y_train, y_pred_train, 'Train Set')
residual_plot(y_test, y_pred_test, 'Test Set')

# Distribution Comparison
distribution_comparison(train, test, 'log_trip_duration', 'Log Trip Duration Distribution')

# Save the best model using joblib
model_dir = os.path.join('D:\\ml_projects\\project-nyc-taxi-trip-duration', 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'ridge_regression_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")
