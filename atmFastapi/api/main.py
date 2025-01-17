import json
import pickle
from enum import Enum
from typing import List

from fastapi import FastAPI
from pandas import DataFrame, concat, read_csv
from pydantic import BaseModel
from redis import StrictRedis
from transform import transform_data

app = FastAPI()
redis_client = StrictRedis(host="redis", port=6379, db=0)


class Pred(BaseModel):
    id_user: int
    atm_group: str
    lat: float
    long: float


class PredBatch(BaseModel):
    id_user: int
    atm_group: List[str]
    lat: List[float]
    long: List[float]


class Feedback(BaseModel):
    id_user: int
    feedback: str


class MethodsType(str, Enum):
    predict = "predict"
    predict_items = "predict_items"
    history = "history"
    feedback = "feedback"


with open("data/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("data/data_pipeline.pkl", "rb") as file:
    data_pipeline = pickle.load(file)


@app.get("/help")
def get_help():
    data = {"methods": ["predict", "predict_items", "history", "feedback"]}
    df = DataFrame(data)
    return df.to_dict(orient="records")


@app.post("/predict")
def predict(pred_body: Pred):
    predict_result = redis_client.get(str(pred_body))
    if predict_result is not None:
        return {"prediction": predict_result.decode()}
    data = DataFrame([pred_body.dict()])
    data_model = data.drop(columns=["id_user"])
    data_model = transform_data(data_model)
    data_model = data_pipeline.transform(data_model)
    pred = model.predict(data_model)
    new_row = {
        "id_user": pred_body.id_user,
        "lat": data["lat"].values[0],
        "long": data["long"].values[0],
        "atm_group": data["atm_group"].values[0],
        "prediction": pred[0],
    }

    data = read_csv("api/predictions.csv")
    data = concat([data, DataFrame([new_row])], ignore_index=True)
    data[["id_user", "lat", "long", "atm_group", "prediction"]].to_csv(
        "api/predictions.csv", index=False
    )
    redis_client.set(str(pred_body), str(pred[0]))
    return {"prediction": str(pred[0])}


@app.post("/predict_batch")
def predict_batch(pred_body: PredBatch):
    predict_batch_result = redis_client.get(str(pred_body))
    if predict_batch_result is not None:
        return json.loads(predict_batch_result)

    data = DataFrame(pred_body.dict())
    model_data = transform_data(data.drop(columns=["id_user"]))
    model_data = data_pipeline.transform(model_data)
    pred = model.predict(model_data)
    data["prediction"] = pred

    old_data = read_csv("api/predictions.csv")
    new_data = concat([old_data, data], ignore_index=True)
    new_data[["id_user", "atm_group", "lat", "long", "prediction"]].to_csv(
        "api/predictions.csv", index=False
    )

    result = data[["lat", "long", "atm_group", "prediction"]].to_dict(orient="records")
    redis_client.set(str(pred_body), json.dumps(result))

    return result


@app.get("/history/{id_user}")
def show_history(id_user: int):
    history_result = redis_client.get(str(id_user))
    if history_result is not None:
        return json.loads(history_result)

    data = read_csv("api/predictions.csv")
    history = data[data["id_user"] == id_user].tail(20).to_dict(orient="records")

    redis_client.set(str(id_user), json.dumps(history))
    return history


@app.post("/feedback")
def save_feedback(feedback_body: Feedback):
    data = read_csv("api/feedback.csv")
    new_row = {"id_user": feedback_body.id_user, "feedback": feedback_body.feedback}
    data = concat([data, DataFrame([new_row])], ignore_index=True)
    data.to_csv("api/feedback.csv", index=False)
    return "Спасибо за отзыв"
