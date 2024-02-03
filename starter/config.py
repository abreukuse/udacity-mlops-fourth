import os

DATA_DIR = "data"
DATA_FILE = "census.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

HYPERPARAMETERS = {
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'criterion': 'gini',
    'class_weight': 'balanced'
}
