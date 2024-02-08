"""
Perform model validation on slices of the data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from textwrap import dedent
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    load_model,
    inference
)
from starter.config import (
    DATA_PATH,
    CATEGORICAL_FEATURES,
    FEATURE_SLICE,
    TARGET_VARIABLE
)


def data_slice_validation():
    """
    Data slice validation.

    This function validates a specific data slice if the specified slice is present in the provided categorical features.
    Data is read from a CSV file, and a trained model is loaded. The data is split into training and testing sets. For each
    unique value in the feature slice, the function processes the data, performs inference using the loaded model, and computes
    model performance metrics. The results are written to an output file 'slice_output.txt'.

    If the feature slice is not specified correctly, a warning is displayed.

    Note:
        Ensure to have properly configured the 'FEATURE_SLICE' and 'CATEGORICAL_FEATURES' variables in 'config.py'.

    Returns:
        None
    """
    if FEATURE_SLICE in CATEGORICAL_FEATURES:
        data = pd.read_csv(DATA_PATH)
        model, encoder, binarizer = load_model()

        train, test = train_test_split(
            data,
            test_size=0.20,
            random_state=42
        )

        feature_unique_values = test[FEATURE_SLICE].unique()

        with open("slice_output.txt", 'w') as f:
            print_variable = f"\tVariable: {FEATURE_SLICE}"
            print(print_variable, file=f)
            print(print_variable)

            for value in feature_unique_values:
                data_slice = test.loc[test[FEATURE_SLICE] == value, :]

                feature_matrix, ground_truth, _, _ = process_data(
                    X=data_slice,
                    categorical_features=CATEGORICAL_FEATURES,
                    label=TARGET_VARIABLE,
                    training=False,
                    encoder=encoder,
                    lb=binarizer
                )

                model_predictions = inference(
                    model=model,
                    X=feature_matrix
                )

                model_performance = compute_model_metrics(
                    y=ground_truth,
                    preds=model_predictions
                )

                precision, recall, fbeta = model_performance

                print_results = dedent(f"""
                Value: {value}
                    - Precision: {precision}
                    - Recall: {recall}
                    - Fbeta: {fbeta}
                """)
                print(print_results, file=f)
                print(print_results)
    else:
        print(
            "Please specify a valid categorical feature in the 'FEATURE_SLICE' variable within the 'config.py' file."
        )


if __name__ == '__main__':
    data_slice_validation()
