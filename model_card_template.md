# Model Card

For additional information see the Model Card paper: [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
- **Model Architecture:** Decision Tree Classifier
- **Hyperparameters:**
  - `max_depth`: 10
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
  - `max_features`: 'sqrt'
  - `criterion`: 'gini'
  - `class_weight`: 'balanced'
  - `random_state`: 42
- **Training Process:** The data was split into training and test sets with an 80-20 split ratio. The decision tree classifier was trained on the training data using the specified hyperparameters.
- **Preprocessing:** Categorical variables were encoded using one-hot encoding to prepare the data for training.

## Intended Use
- **Purpose:** Predict whether a person's income exceeds $50,000 per year based on census data.
- **Intended Users:** Researchers, policymakers, and organizations interested in socioeconomic analysis and targeted interventions based on income level predictions.

## Training Data
- **Dataset:** Records extracted from the 1994 Census database by Barry Becker.
- **Preprocessing:** One-hot encoding applied to categorical variables.
- **Features:** Categorical and integer features including demographic and employment-related attributes.

## Evaluation Data
- **Dataset:** Split into training and test sets with an 80-20 ratio.

## Metrics
- **Performance Metrics:** Precision, Recall, F-beta Score
  - **Precision:** 0.50
  - **Recall:** 0.86
  - **F-beta Score:** 0.63

## Ethical Considerations
- **Bias and Fairness:** The model's predictions may be influenced by biases present in the training data, potentially leading to disparities in prediction accuracy across different demographic groups.
- **Privacy Concerns:** The model's predictions may reveal sensitive personal information about individuals, raising privacy concerns regarding the use and dissemination of the model's outputs.

## Caveats and Recommendations
- **Limitations:** The model's predictions are based solely on the features available in the census data and may not capture all factors influencing income level.
- **Recommendations:** Exercise caution when interpreting and using the model's predictions, considering its limitations and potential biases. Regularly monitor and update the model to account for changes in socioeconomic dynamics.
