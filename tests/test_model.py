"""
Test fucntions in the starter.ml.model module
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from starter.config import (
    HYPERPARAMETERS,
    MODEL_DIR
)
from starter.ml.model import (
    train_model,
    compute_model_metrics,
    load_model,
    inference
)


@pytest.fixture
def model():
    model_, encoder_ = load_model(
        model_dir=MODEL_DIR
    )

    return model_

def test_train_model():
    """
    Test 'train_model' function, expecting a DecisionTreeClassifier.
    """

    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(2, size=(50, 1))
    
    model = train_model(
        X_train,
        y_train,
        HYPERPARAMETERS
    )

    assert isinstance(model, DecisionTreeClassifier)


def test_compute_model_metrics():
    """
    Test compute_model_metrics with perfect predictions, expecting precision, recall, and fbeta to be 1 and of type float.
    """

    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(
        y_true,
        y_pred
    )

    assert precision == 1
    assert recall == 1
    assert fbeta == 1

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(model):
    """
    Test if the returned value of the inference function is a numpy array.
    """

    X = np.random.rand(50, 108)
    preds = inference(
        model=model,
        X=X
    )

    assert isinstance(preds, np.ndarray)
    assert len(preds) == 50
