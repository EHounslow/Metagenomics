from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from data_loader import extract_abundance_data
from variable_definer import get_variables
from disease_selector import get_disease_data
from modellers.random_forest_modeller import run_random_forest
from modellers.logistic_regression_modeller import run_logistic_regression
from dataset_selector import get_dataset

#read in data
abundance = "data/abundance.txt"
abundance_data = extract_abundance_data(abundance)
data_subset = get_dataset("Quin_gut_liver_cirrhosis", abundance_data)



# Function to select the disease to analyse
combined_disease_control, disease_target, control = get_disease_data("cirrhosis", data_subset)



# Create a list of taxonomic and metadata variables
species, taxonomy, metadata = get_variables(combined_disease_control)



# Define the predictive features
x = combined_disease_control[species]
# Define the target variable, disease
y = combined_disease_control["disease"]

# Split data into training and validation data, for both features and target
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)




# Define logistic regression model
logistic_reg_model = run_logistic_regression(x, y)

disease_x = disease_target[species]
disease_y = disease_target["disease"]

# Define decision tree regressor model. Specify a number for random_state to ensure same results each run
#disease_model = DecisionTreeRegressor(random_state=1)


disease_predictions = logistic_reg_model.predict(disease_x)

# Compare the predictions with the actual values by looking at the first few results
#print(preds_val[:10])
#print(val_y[:10])

#print(mean_absolute_error(val_y, preds_val))

# Define function that retrieves the mean absolute error based on different numbers of maximum leaf nodes
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Create a dictionary containing a range of leaf node numbers with their mae values
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

mae_values = {}

for node in candidate_max_leaf_nodes:
    mae = get_mae(node, train_x, val_x, train_y, val_y)
    mae_values[node] = mae

# Retrieve the optimum tree size with the lowest mae
best_tree_size = min(mae_values, key=mae_values.get)

# Define the refined model
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 0)

# Now that model has been refined, use all the data to fit it
final_model.fit(x, y)

run_random_forest(train_x, train_y, val_x, val_y)
