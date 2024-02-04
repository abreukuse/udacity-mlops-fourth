import os

DATA_DIR = "data"
DATA_FILE = "census.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

MODEL_DIR = "model"
MODEL_FILE = "model.pkl"
ENCODER_FILE = "encoder.pkl"
BINARIZER_FILE = "binarizer.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
ENCODER_PATH = os.path.join(MODEL_DIR, ENCODER_FILE)
BINARIZER_PATH = os.path.join(MODEL_DIR, BINARIZER_FILE)

HYPERPARAMETERS = {
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'criterion': 'gini',
    'class_weight': 'balanced',
    'random_state': 42
}

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

FEATURE_SLICE = "education"
TARGET_VARIABLE = "salary"
