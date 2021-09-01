from sklearn.model_selection import train_test_split

from data_loader import extract_abundance_data
from dataset_selector import get_dataset
from variable_definer import get_variables
from disease_selector import get_disease_data
from modellers.random_forest_modeller import run_random_forest
from modellers.logistic_regression_modeller import run_logistic_regression
from modellers.decision_tree_regressor import run_decision_tree_regressor

# Read in data
abundance_data = extract_abundance_data("data/abundance.txt")
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
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

# Define logistic regression model
logistic_reg_model = run_logistic_regression(x, y)

disease_x = disease_target[species]
disease_y = disease_target["disease"]
disease_predictions = logistic_reg_model.predict(disease_x)

run_decision_tree_regressor(train_x, val_x, train_y, val_y, x, y)
run_random_forest(train_x, train_y, val_x, val_y)
