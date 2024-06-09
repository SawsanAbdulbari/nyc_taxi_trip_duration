from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Define root_mean_squared_error as a standalone function
def root_mean_squared_error(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).
    
    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    
    Returns:
    float: RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Evaluation Functions
def predict_eval(model, data, target, name):
    """
    Evaluates the model's performance on a given dataset.
    
    Parameters:
    model: Trained model.
    data (pd.DataFrame or np.ndarray): Input features.
    target (pd.Series or np.ndarray): True target values.
    name (str): Name of the dataset (for display purposes).
    
    Returns:
    None
    """
    y_pred = model.predict(data)
    rmse = root_mean_squared_error(target, y_pred)
    r2 = r2_score(target, y_pred)
    print(f"{name} Evaluation:\n  RMSE: {rmse:.4f}\n  R²: {r2:.4f}")

def cross_val_eval(model, X, y, cv=5):
    """
    Evaluates the model using cross-validation.
    
    Parameters:
    model: Model to be evaluated.
    X (pd.DataFrame or np.ndarray): Features for cross-validation.
    y (pd.Series or np.ndarray): Target for cross-validation.
    cv (int): Number of cross-validation folds.
    
    Returns:
    tuple: Mean RMSE and R² scores from cross-validation.
    """
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv)
    mean_rmse = -scores.mean()
    mean_r2 = r2_scores.mean()
    return mean_rmse, mean_r2


# Residual Plot
def residual_plot(y_true, y_pred, name):
    """
    Plots the residuals of the predictions.
    
    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    name (str): Name of the dataset (for display purposes).
    
    Returns:
    None
    """
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
    """
    Compares the distribution of a feature between train and test datasets.
    
    Parameters:
    train (pd.DataFrame): Training dataset.
    test (pd.DataFrame): Test dataset.
    feature (str): Feature column to compare.
    title (str): Title for the plot.
    
    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train[feature], label='Train', shade=True, color='blue')
    sns.kdeplot(test[feature], label='Test', shade=True, color='orange')
    plt.title(title)
    plt.legend()
    plt.show()
