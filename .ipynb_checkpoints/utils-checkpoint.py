import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import make_pipeline
from linearmodels.panel import PanelOLS
import statsmodels.api as sm


model_performance = []

# Helper function to store models
def store_model_performance(y_true, y_pred, model_name):
    """
    Function to compute MSE and R-squared for given predictions and actual values,
    and store these metrics in a list of dictionaries.

    Args:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values from the model.
    model_name (str): Name of the model.

    Returns:
    None: Appends the performance metrics to the global list 'model_performance'.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    performance_dict = {
        'model_name': model_name,
        'mse': mse,
        'r2': r2
    }

    model_performance.append(performance_dict)