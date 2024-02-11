"""wip
"""

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true: np.array, y_pred: np.array, features_shape: tuple
) -> dict:
    """
    This function calculates various regression metrics given true and predicted values.

    Args:
        y_true (np.array): Ground truth (correct) target values.
        y_pred (np.array): Estimated target values.
        features_shape (tuple): Shape of the feature in (row, column) format

    Returns:
        dict: A dictionary of regression metrics, including:
            - MAE (float): Mean Absolute Error
            - MSE (float): Mean Squared Error
            - RMSE (float): Root Mean Squared Error
            - MPE (float): Mean Percentage Error
            - MAPE (float): Mean Absolute Percentage Error
            - R2-Score (float): Coefficient of determination
            - Adjusted R2-Score (float): Accounts for the number of predictors
            - Explained Variance Score (float): Explains the variance
    """

    # get MAE, MSE and R2 values
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # calculate: Mean Percentage Error(MPE)
    mpe = np.mean((y_true - y_pred) / y_true) * 100

    # calculate: Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # get number of observations (n)
    n = features_shape[0]

    # get number of predictors/features (p)
    p = features_shape[1]

    # calculate adjusted-R2
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    # Explained variance score
    expl_var_score = explained_variance_score(y_true, y_pred)

    regression_metrics_dict = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MPE": round(mpe, 2),
        "MAPE": round(mape, 2),
        "R2-Score": round(r2, 2),
        "Adjusted R2-Score": round(adj_r2, 2),
        "Explained Variance Score": round(expl_var_score, 2),
    }

    return regression_metrics_dict
