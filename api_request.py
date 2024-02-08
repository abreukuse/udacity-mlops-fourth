"""
Perform a request to the API for each prediction.
"""


import requests
from textwrap import dedent


lower_50k_input = {
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

higher_50k_input = {
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


def get_api_response(url, model_input):
    response = requests.post(
        url=url,
        json=model_input
    )

    prompt_message = dedent(f"""
        STATUS CODE: {response.status_code}
        API OUTPUT: {response.json()}
        """)
    print(prompt_message)


def main():
    api_endpont = "https://udacity-mlops-fourth-project.onrender.com/inference/"
    get_api_response(
        url=api_endpont,
        model_input=lower_50k_input
    )
    get_api_response(
        url=api_endpont,
        model_input=higher_50k_input
    )


if __name__ == '__main__':
    main()
