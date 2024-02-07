# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from starter.ml.model import load_model
from starter.ml.data import process_data
from starter.config import (
    TARGET_VARIABLE,
    CATEGORICAL_FEATURES
)


class ModelInputs(BaseModel):
    age: int = Field(
        examples=[39]
    )
    workclass: str = Field(
        examples=["State-gov", "Private", "Self-emp-not-inc"]
    )
    fnlgt: int = Field(
        examples=[77516]
    )
    education: str = Field(
        examples=["Bachelors", "HS-grad", "Masters"]
    )
    education_num: int = Field(
        alias="education-num",
        examples=[13]
    )
    marital_status: str = Field(
        alias="marital-status",
        examples=["Never-married", "Married-civ-spouse", "Divorced"]
    )
    occupation: str = Field(
        examples=["Adm-clerical", "Craft-repair", "Prof-specialty"]
    )
    relationship: str = Field(
        examples=["Not-in-family", "Husband", "Wife"]
    )
    race: str = Field(
        examples=["White", "Black", "Asian-Pac-Islander"]
    )
    sex: str = Field(
        examples=["Male", "Female"]
    )
    capital_gain: int = Field(
        alias="capital-gain",
        examples=[2174]
    )
    capital_loss: int = Field(
        alias="capital-loss",
        examples=[0]
    )
    hours_per_week: int = Field(
        alias="hours-per-week",
        examples=[40]
    )
    native_country: str = Field(
        alias="native-country",
        examples=["United-States", "India"]
    )


app = FastAPI()

model, encoder, binarizer = load_model()


@app.get("/")
async def say_hello():
    return {
        "ML Model": "Welcome!"
    }


@app.post("/inference/")
async def model_inferece(features: ModelInputs):
    df_input = pd.DataFrame(features.dict(by_alias=True), index=[0])
    feature_vector, _, _, _ = process_data(
        X=df_input,
        categorical_features=CATEGORICAL_FEATURES,
        training=False,
        encoder=encoder
    )

    prediction = model.predict(feature_vector)
    predicted_label = binarizer.inverse_transform(prediction)[0]

    return {TARGET_VARIABLE: predicted_label}
