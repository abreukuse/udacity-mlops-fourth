"""
Perform model validation on slices of the data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
	compute_model_metrics,
	load_model,
	inference
)
from starter.config import (
	DATA_PATH,
	MODEL_PATH,
	ENCODER_PATH,
	CATEGORICAL_FEATURES,
	FEATURE_SLICE,
	TARGET_VARIABLE
)


def data_slice_validation():
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
			data_slice = test.loc[test[FEATURE_SLICE]==value, :]

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

			print_results = f"""
			Value: {value}
				- Precision: {precision}
				- Recall: {recall}
				- Fbeta: {fbeta}
			"""
			print(print_results, file=f)
			print(print_results)


if __name__ == '__main__':
	data_slice_validation()
