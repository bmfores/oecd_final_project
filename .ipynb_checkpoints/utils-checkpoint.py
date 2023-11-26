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