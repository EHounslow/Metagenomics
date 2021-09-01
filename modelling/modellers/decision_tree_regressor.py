from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def get_mean_absolute_error(max_leaf_nodes, train_x, val_x, train_y, val_y):
    """
    A function that retrieves the mean absolute error based on
    different numbers of maximum leaf nodes
    """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


def run_decision_tree_regressor(train_x, val_x, train_y, val_y, x, y):
    """ """
    # Create a dictionary containing a range of leaf node numbers with their mae values
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

    mae_values = {}

    for node in candidate_max_leaf_nodes:
        mae = get_mean_absolute_error(node, train_x, val_x, train_y, val_y)
        mae_values[node] = mae

    # Retrieve the optimum tree size with the lowest mae
    best_tree_size = min(mae_values, key=mae_values.get)

    # Define decision tree regressor model.
    # Specify a number for random_state to ensure same results each run
    final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

    # disease_model = DecisionTreeRegressor(random_state=1)

    # Now that model has been refined, use all the data to fit it
    final_model.fit(x, y)
