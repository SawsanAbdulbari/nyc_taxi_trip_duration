import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score

def build_pipeline(categorical_features, numeric_features):
    """
    Builds a preprocessing and modeling pipeline with a ColumnTransformer and Ridge regression.

    Parameters:
    categorical_features (list of str): List of categorical feature names.
    numeric_features (list of str): List of numeric feature names.

    Returns:
    Pipeline: A scikit-learn pipeline with preprocessing and Ridge regression.
    """
    # Create the Column Transformer
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features),
        ('log_transform', FunctionTransformer(np.log1p, validate=True), ['distance_haversine', 'distance_manhattan', 'direction'])
    ], remainder='passthrough')
    
    # Create the Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('regression', Ridge(alpha=1.0))
    ])
    
    return pipeline

def train_model(pipeline, train_features, train_target):
    """
    Trains the pipeline model using the provided training data.

    Parameters:
    pipeline (Pipeline): The machine learning pipeline.
    train_features (pd.DataFrame): Training features.
    train_target (pd.Series): Training target.

    Returns:
    Pipeline: The trained model pipeline.
    """
    model = pipeline.fit(train_features, train_target)
    return model

def cross_val_eval(pipeline, X, y, cv=5):
    """
    Evaluates the pipeline model using cross-validation.

    Parameters:
    pipeline (Pipeline): The machine learning pipeline.
    X (pd.DataFrame or np.ndarray): Features for cross-validation.
    y (pd.Series or np.ndarray): Target for cross-validation.
    cv (int): Number of cross-validation folds.

    Returns:
    None
    """
    rmse_scores = cross_val_score(pipeline, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(pipeline, X, y, scoring='r2', cv=cv)
    print(f"Cross-Validation RMSE: {-rmse_scores.mean():.4f}, R²: {r2_scores.mean():.4f}")
