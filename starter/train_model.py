# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    save_model,
    compute_model_metrics,
    inference
)
from config import (
    DATA_PATH,
    HYPERPARAMETERS,
    CATEGORICAL_FEATURES,
    TARGET_VARIABLE
)

# Add code to load in the data.
data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=CATEGORICAL_FEATURES,
    label=TARGET_VARIABLE,
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=CATEGORICAL_FEATURES,
    label=TARGET_VARIABLE,
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(
    X_train=X_train,
    y_train=y_train,
    hyperparameters=HYPERPARAMETERS
)

y_pred = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(
    y=y_test,
    preds=y_pred
)

print(
    f"""
Model performance on test data.

    - Precision: {round(precision, 2)}
    - Recall: {round(recall, 2)}
    - Fbeta: {round(fbeta, 2)}
"""
)

save_model(
    model=model,
    encoder=encoder,
    lb=lb
)
