from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json() == {"ML Model": "Welcome!"}


def test_inferece_lower_50k():
    input_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/inference/", json=input_data)

    assert response.status_code == 200
    assert response.json() == {"salary": "<=50K"}


def test_inference_higher_50k():
    input_data = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 20000,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    response = client.post("/inference/", json=input_data)

    assert response.status_code == 200
    assert response.json() == {"salary": ">50K"}
