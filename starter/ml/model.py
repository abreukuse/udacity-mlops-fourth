import os
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, hyperparameters):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    hyperparameters : Dict
        Model hyperparameters
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = DecisionTreeClassifier(
        **hyperparameters
    )
    model.fit(
        X=X_train,
        y=y_train
    )

    return model


def save_model(model, encoder, output_path):
    """
    Save a machine learning model and categorical encoder using pickle.

    Inputs
    ------
    - model : sklearn.tree._classes.DecisionTreeClassifier
        The machine learning model to be saved.
    - encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    - output_path : str
        The file path where the model will be saved.

    Returns
    -------
    None

    """
    model_path = os.path.join(output_path, 'model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    encoder_path = os.path.join(output_path, 'encoder.pkl')
    with open(encoder_path, 'wb') as encoder_file:
        pickle.dump(encoder, encoder_file)


def load_model(model_dir):
    """
    Load a machine learning model and an encoder from the specified output path.

    Parameters:
    - model_dir (str): The directory path containing the saved model and encoder files.

    Returns:
    tuple: A tuple containing the loaded machine learning model and encoder.

    """
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    encoder_path = os.path.join(model_dir, 'encoder.pkl')
    with open(encoder_path, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)

    return model, encoder


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
