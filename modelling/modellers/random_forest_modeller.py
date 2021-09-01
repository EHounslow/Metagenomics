from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pandas.core.frame import DataFrame


def run_random_forest(
    train_x: DataFrame, train_y: DataFrame, val_x: DataFrame, val_y: DataFrame
) -> None:
    """
    Runs a random forest model.
    Args:
        train_x: (DataFrame): Training data for x.
        train_y: (DataFrame): Training data for y.
        val_x: (DataFrame):  Validation data for x.
        val_y: (DataFrame): Validation data for y.

    Returns:
        Printed statement
    """

    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(random_state=1)

    # Fit model
    rf_model.fit(train_x, train_y)

    # Calculate the mean absolute error of your Random Forest model on the validation data
    rf_val_predictions = rf_model.predict(val_x)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    return print(f"Validation MAE for Random Forest Model: {rf_val_mae}")
