# REPO LINK: https://github.com/abreukuse/udacity-mlops-fourth


Train and deploy a machine learning model to predict whether a person's income exceeds $50,000 per year based on census data.

- Requirements
python 3.8 and the dependencies in the requirements.txt file.

### How to use it
- Set the PYTHONPATH on the root directory of the project
`export PYTHONPATH="$PWD:$PYTHONPATH"`

- Install the dependencies 
`pip install -r requirements.txt`

- Train the model
`pyhton3 starter/train_model.py`

- Deploy it locally
`uvicorn main:app --reload`

- Run tests
`pytest -v .`

