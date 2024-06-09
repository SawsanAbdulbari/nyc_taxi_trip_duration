import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Distribution Comparison
def distribution_comparison(train, test, feature, title):
    """
    Compare the distribution of a feature between train and test datasets.
    
    Parameters:
    train (pd.DataFrame): Training dataset.
    test (pd.DataFrame): Test dataset.
    feature (str): Feature column to compare.
    title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train[feature], label='Train', shade=True, color='blue')
    sns.kdeplot(test[feature], label='Test', shade=True, color='orange')
    plt.title(title)
    plt.legend()
    plt.show()

# Helper Functions for Geographic Features
def distance(lat1, lon1, lat2, lon2, unit='km'):
    """
    Calculate the distance between two geographic coordinates.
    
    Parameters:
    lat1, lon1, lat2, lon2 (float): Latitude and longitude of the two points.
    unit (str): Unit of distance ('km' or 'miles').
    
    Returns:
    float: Distance between the points.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 if unit == 'km' else 3956
    return c * r

def manhattan_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the Manhattan distance between two geographic coordinates.
    
    Parameters:
    lat1, lng1, lat2, lng2 (float): Latitude and longitude of the two points.
    
    Returns:
    float: Manhattan distance between the points.
    """
    horizontal_distance = distance(lat1, lng1, lat1, lng2)
    vertical_distance = distance(lat1, lng1, lat2, lng1)
    return horizontal_distance + vertical_distance

def bearing_array(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two geographic coordinates.
    
    Parameters:
    lat1, lon1, lat2, lon2 (float): Latitude and longitude of the two points.
    
    Returns:
    float: Bearing in degrees.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    return (bearing + 360) % 360

# Feature Engineering Functions
def extract_datetime_features(df):
    """
    Extract datetime features from the 'pickup_datetime' column.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing the 'pickup_datetime' column.
    
    Returns:
    pd.DataFrame: Dataframe with additional datetime features.
    """
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    return df

def remove_outliers(train, column, n_std=2):
    """
    Remove outliers from a specified column based on standard deviation.
    
    Parameters:
    train (pd.DataFrame): Training dataset.
    column (str): Column from which to remove outliers.
    n_std (int): Number of standard deviations to use for the threshold.
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed.
    """
    m = np.mean(train[column])
    s = np.std(train[column])
    lower_bound = m - n_std * s
    upper_bound = m + n_std * s
    return train[(train[column] >= lower_bound) & (train[column] <= upper_bound)]

def create_airport_features(train):
    """
    Create features indicating proximity to airports.
    
    Parameters:
    train (pd.DataFrame): Training dataset.
    
    Returns:
    pd.DataFrame: Dataframe with new airport-related features.
    """
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
    """
    Filter rows based on geographic bounds.
    
    Parameters:
    df (pd.DataFrame): Dataframe to filter.
    longitude_min, longitude_max (float): Longitude bounds.
    latitude_min, latitude_max (float): Latitude bounds.
    
    Returns:
    pd.DataFrame: Filtered dataframe.
    """
    df = df[(df['pickup_longitude'] >= longitude_min) & (df['pickup_longitude'] <= longitude_max)]
    df = df[(df['pickup_latitude'] >= latitude_min) & (df['pickup_latitude'] <= latitude_max)]

    # Filter based on drop-off coordinates
    df = df[(df['dropoff_longitude'] >= longitude_min) & (df['dropoff_longitude'] <= longitude_max)]
    df = df[(df['dropoff_latitude'] >= latitude_min) & (df['dropoff_latitude'] <= latitude_max)]

    return df