import pandas as pd
import numpy as np
import os
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# Define root_mean_squared_error as a standalone function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Evaluation Functions
def predict_eval(model, data, target, name):
    y_pred = model.predict(data)
    rmse = root_mean_squared_error(target, y_pred)
    r2 = r2_score(target, y_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def cross_val_eval(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv)
    print(f"Cross-Validation RMSE: {-scores.mean():.4f}, R²: {r2_scores.mean():.4f}")

# Residual Plot
def residual_plot(y_true, y_pred, name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot: {name}')
    plt.show()

# Distribution Comparison
def distribution_comparison(train, test, feature, title):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train[feature], label='Train', shade=True, color='blue')
    sns.kdeplot(test[feature], label='Test', shade=True, color='orange')
    plt.title(title)
    plt.legend()
    plt.show()

# Helper Functions for Geographic Features
def distance(lat1, lon1, lat2, lon2, unit='km'):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 if unit == 'km' else 3956
    return c * r

def manhattan_distance(lat1, lng1, lat2, lng2):
    horizontal_distance = distance(lat1, lng1, lat1, lng2)
    vertical_distance = distance(lat1, lng1, lat2, lng1)
    return horizontal_distance + vertical_distance

def bearing_array(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    return (bearing + 360) % 360

# Feature Engineering Functions
def extract_datetime_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    return df

def remove_outliers(train, column, n_std=2):
    m = np.mean(train[column])
    s = np.std(train[column])
    lower_bound = m - n_std * s
    upper_bound = m + n_std * s
    return train[(train[column] >= lower_bound) & (train[column] <= upper_bound)]

def create_airport_features(train):
    jfk_bounds = (-73.8352, -73.7401, 40.6195, 40.6659)
    lga_bounds = (-73.8895, -73.8522, 40.7664, 40.7931)
    ewr_bounds = (-74.1925, -74.1594, 40.6700, 40.7081)

    def in_bounds(lat, lon, bounds):
        lon_min, lon_max, lat_min, lat_max = bounds
        return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

    train['pickup_jfk'] = train.apply(lambda x: in_bounds(x['pickup_latitude'], x['pickup_longitude'], jfk_bounds), axis=1)
    train['pickup_lga'] = train.apply(lambda x: in_bounds(x['pickup_latitude'], x['pickup_longitude'], lga_bounds), axis=1)
    train['pickup_ewr'] = train.apply(lambda x: in_bounds(x['pickup_latitude'], x['pickup_longitude'], ewr_bounds), axis=1)

    train['dropoff_jfk'] = train.apply(lambda x: in_bounds(x['dropoff_latitude'], x['dropoff_longitude'], jfk_bounds), axis=1)
    train['dropoff_lga'] = train.apply(lambda x: in_bounds(x['dropoff_latitude'], x['dropoff_longitude'], lga_bounds), axis=1)
    train['dropoff_ewr'] = train.apply(lambda x: in_bounds(x['dropoff_latitude'], x['dropoff_longitude'], ewr_bounds), axis=1)
    return train
# Define the filter_geographic_bounds function separately
def filter_geographic_bounds(df, longitude_min, longitude_max, latitude_min, latitude_max):
    # Filter based on pickup coordinates
    df = df[(df['pickup_longitude'] >= longitude_min) & (df['pickup_longitude'] <= longitude_max)]
    df = df[(df['pickup_latitude'] >= latitude_min) & (df['pickup_latitude'] <= latitude_max)]

    # Filter based on drop-off coordinates
    df = df[(df['dropoff_longitude'] >= longitude_min) & (df['dropoff_longitude'] <= longitude_max)]
    df = df[(df['dropoff_latitude'] >= latitude_min) & (df['dropoff_latitude'] <= latitude_max)]

    return df

# Approach 1 - Ridge Regression with Direct Features
def approach1(train, test):
    numeric_features = ['pickup_latitude',
                        'pickup_longitude',
                        'dropoff_latitude',
                        'dropoff_longitude'
                        ]

    categorical_features = ['dayofweek', 
                            'month', 
                            'hour', 
                            'day', 
                            'vendor_id',
                            'passenger_count', 
                            'store_and_fwd_flag']

    train_features = numeric_features + categorical_features

    # For Debug
    missing_features = [feat for feat in train_features if feat not in train.columns]
    if missing_features:
        raise ValueError(f"Missing features in the train set: {missing_features}")

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
    ], 
    remainder='passthrough'
)

    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('regression', Ridge(alpha=1.0))
    ])

    model = pipeline.fit(train[train_features], train['log_trip_duration'])

    predict_eval(model, train[train_features], train['log_trip_duration'], "train")
    predict_eval(model, test[train_features], test['log_trip_duration'], "test")

    cross_val_eval(pipeline, train[train_features], train['log_trip_duration'])

# Approach 2 - One-Hot Encoding and Ridge Regression
def approach2(train, test):
    # One-Hot Encoding of Categorical Features
    store_and_fwd_flag_train = pd.get_dummies(train.store_and_fwd_flag, prefix='sf', prefix_sep='_')
    store_and_fwd_flag_test = pd.get_dummies(test.store_and_fwd_flag, prefix='sf', prefix_sep='_')

    month_train = pd.get_dummies(train.month, prefix='m', prefix_sep='_')
    month_test = pd.get_dummies(test.month, prefix='m', prefix_sep='_')

    dow_train = pd.get_dummies(train.dayofweek, prefix='dow', prefix_sep='_')
    dow_test = pd.get_dummies(test.dayofweek, prefix='dow', prefix_sep='_')

    hour_train = pd.get_dummies(train.hour, prefix='h', prefix_sep='_')
    hour_test = pd.get_dummies(test.hour, prefix='h', prefix_sep='_')

    # Keep `log_trip_duration` for training
    target_train = train['log_trip_duration']
    target_test = test['log_trip_duration']

    # Drop Unwanted Columns (but not `log_trip_duration`)
    train = train.drop(['vendor_id', 
                        'passenger_count', 
                        'store_and_fwd_flag', 
                        'month', 'hour', 
                        'dayofweek',
                        'pickup_longitude', 
                        'pickup_latitude', 
                        'dropoff_longitude', 
                        'dropoff_latitude',
                        'trip_duration',
                        'log_trip_duration'
                        ], axis=1, errors='ignore')
    test = test.drop([
                    'vendor_id', 
                    'passenger_count', 
                    'store_and_fwd_flag', 
                    'month', 
                    'hour', 
                    'dayofweek',
                    'pickup_longitude', 
                    'pickup_latitude', 
                    'dropoff_longitude', 
                    'dropoff_latitude',
                    'trip_duration', 
                    'log_trip_duration'], axis=1, errors='ignore')

    # Combine One-Hot Encoded Columns
    train_main = pd.concat([train,
                            store_and_fwd_flag_train,
                            month_train,
                            hour_train,
                            dow_train], axis=1)

    test_main = pd.concat([test,
                            store_and_fwd_flag_test,
                            month_test,
                            hour_test,
                            dow_test], axis=1)

    # Reindex Validation Dataset to Match Training Dataset
    test_main = test_main.reindex(columns=train_main.columns, fill_value=0)

    # Define Numeric Features
    numeric_features = ['distance_haversine', 
                        'distance_manhattan', 
                        'direction']

    # Create Feature List without `log_trip_duration`
    all_features = train_main.columns.tolist()

    numeric_transformer = StandardScaler()
    column_transformer = ColumnTransformer([
        ('scaling', numeric_transformer, numeric_features)
    ], remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('regression', Ridge())
    ])

    # Train and Evaluate Model
    model = pipeline.fit(train_main[all_features], target_train)

    predict_eval(model, train_main[all_features], target_train, "train")
    predict_eval(model, test_main[all_features], target_test, "test")

    cross_val_eval(pipeline, train_main[all_features], target_train)

# Approach 3 - Improved Ridge Regression Model with Residual Plots
def approach3(train, test):
    numeric_features = ['pickup_latitude', 
                        'pickup_longitude',
                        'dropoff_latitude', 
                        'dropoff_longitude',
                        'distance_haversine', 
                        'distance_manhattan', 
                        'direction',
                        'trip_speed'
                        ]

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
                            'dropoff_ewr'
                            ]

    train_features = numeric_features + categorical_features

    missing_features = [feat for feat in train_features if feat not in train.columns]
    if missing_features:
        raise ValueError(f"Missing features in the train set: {missing_features}")
    
    # Create the Column Transformer
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features),
        ('log_transform', FunctionTransformer(np.log1p, validate=True),[
            'distance_haversine', 'distance_manhattan', 'direction'
            ])
    ], remainder='passthrough')
    
    # Create the Pipeline
    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge(alpha=1.0))
    ])

    # Train the model
    model = pipeline.fit(train[train_features], train['log_trip_duration'])
    
    # Predictions
    y_pred_train = model.predict(train[train_features])
    y_pred_test = model.predict(test[train_features])

    
    # Evaluate the model
    predict_eval(model, train[train_features], train['log_trip_duration'], "train")
    predict_eval(model, test[train_features], test['log_trip_duration'], "test")
    cross_val_eval(model, train[train_features], train['log_trip_duration'])
    

    # Save the best model using joblib
    model_path = os.path.join('project-nyc-taxi-trip-duration', 'ridge_regression_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Residual Plots
    residual_plot(train['log_trip_duration'], y_pred_train, 'Train Set')
    residual_plot(test['log_trip_duration'], y_pred_test, 'Test Set')

    # Distribution Comparison Plots
    for feature in numeric_features + ['log_trip_duration']:
        distribution_comparison(train, test, feature, f'Distribution Comparison: {feature}')


def prepare_data(train):
    """Prepares the data by adding new features and calculating distances/directions."""

    train = extract_datetime_features(train)
    # Defining bins for the hours of the day
    # bins = [0, 6, 9, 12, 15, 18, 21, 24]
    # labels = ['Early Morning', 'Morning', 'Noon', 'Afternoon', 'Evening', 'Night','Late Night']

    # # Creating a new column 'time_period' based on the hour bins
    # train['time_period'] = pd.cut(train['hour'], bins=bins, labels=labels, right=False)
    # Filter based on geographic bounds
    longitude_min, longitude_max = -74.2591, -73.7004
    latitude_min, latitude_max = 40.4774, 40.9176
    
    train = filter_geographic_bounds(train, longitude_min, longitude_max, latitude_min, latitude_max)

    train['distance_haversine'] = distance(train['pickup_latitude'],
                                    train['pickup_longitude'],
                                    train['dropoff_latitude'],
                                    train['dropoff_longitude'])

    train['distance_manhattan'] = manhattan_distance(train['pickup_latitude'],
                                                train['pickup_longitude'],
                                                train['dropoff_latitude'],
                                                train['dropoff_longitude'])

    train['direction'] = bearing_array(train['pickup_latitude'],
                                train['pickup_longitude'],
                                train['dropoff_latitude'],
                                train['dropoff_longitude'])
    
    # Create airport features
    train = create_airport_features(train)
    
    train['trip_speed'] = train['distance_haversine'] / (train['trip_duration'] / 3600)

    train['log_trip_duration'] = np.log1p(train['trip_duration'])

    train = train.drop(columns=['id', 
                                'pickup_datetime', 
                                'trip_duration'], errors='ignore')
    
    train = remove_outliers(train, 'log_trip_duration', n_std=2)

    return train
if __name__ == '__main__':
    root_dir = 'D:\\ml_projects\\project-nyc-taxi-trip-duration\\data\\split'
    train = pd.read_csv(os.path.join(root_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(root_dir, 'val.csv'))

    train = prepare_data(train)
    test = prepare_data(test)

    print("Approach 1 : ")
    approach1(train, test)
    print("Approach 2 : ")
    approach2(train, test)
    print("Approach 3 : ")
    approach3(train, test)
