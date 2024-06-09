# feature_engineering.py
import numpy as np
# from utils import (
#     extract_datetime_features,
#     filter_geographic_bounds,
#     distance,
#     manhattan_distance,
#     bearing_array,
#     create_airport_features,
#     remove_outliers
# )
from utils import bearing_array, create_airport_features, distance, extract_datetime_features, filter_geographic_bounds, manhattan_distance, remove_outliers


def prepare_data(train):
    """
    Prepares the data by adding new features and calculating distances/directions.

    Parameters:
    train (pd.DataFrame): The training dataset.

    Returns:
    pd.DataFrame: The prepared dataset with new features.
    """
    # Extract datetime features
    train = extract_datetime_features(train)
    
    # Filter based on geographic bounds
    longitude_min, longitude_max = -74.2591, -73.7004
    latitude_min, latitude_max = 40.4774, 40.9176
    train = filter_geographic_bounds(train, longitude_min, longitude_max, latitude_min, latitude_max)

    # Calculate distances and directions
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
    
    # Calculate trip speed and log of trip duration
    train['trip_speed'] = train['distance_haversine'] / (train['trip_duration'] / 3600)

    train['log_trip_duration'] = np.log1p(train['trip_duration'])
    
    # Drop unnecessary columns
    train = train.drop(columns=['id', 
                                'pickup_datetime', 
                                'trip_duration'], errors='ignore')
    
    # Remove outliers based on log_trip_duration
    train = remove_outliers(train, 'log_trip_duration', n_std=2)

    return train
