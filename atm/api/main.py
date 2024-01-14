from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pandas import DataFrame, read_csv
from io import StringIO
from ..preprocessing.data_manipulation import preprocess
from ..data import read_data
from .. import models_dir_local
from enum import Enum
app = FastAPI()

class Pred(BaseModel):
    id_user: int
    lat: float
    long: float

class Feedback(BaseModel):
    id_user: int
    feedback: str

class MethodsType(str, Enum):
    predict = "predict"
    predict_items = "predict_items"
    history = "history"
    feedback = "feedback"

model = read_data(models_dir_local)
@app.get("/help")
def Help():
    data = {'methods': ['predict', 'predict_items', 'history', 'feedback']}
    df = DataFrame(data)
    return df.to_dict(orient="records")

@app.post("/methods")
def Methods(like : MethodsType):
    data = {'methods': ['predict', 'predict_items', 'history', 'feedback'], 'description': ['Укажите координаты через , Пример: 54.68,55.69',
                                                                                            'Загрузите файл с колонками lat и long.',
                                                                                            'История Ваших запросов',
                                                                                            'Вы можете оставить обратную связь']}
    df = DataFrame(data)
    if like == 'predict_items':
        response = FileResponse('atm/api/example.csv')
    elif like in ('predict', 'history', 'feedback'):
        response = df[df["methods"] == like].to_dict(orient="records")
    return response
@app.post("/predict")
def Predict(pred_body : Pred):
    data = DataFrame([pred_body.dict()])
    data = data.drop(columns=["id_user"])
    data = preprocess(data)
    pred = model.predict(data)
    data = read_csv('atm/api/predictions.csv')
    new_row = {'id_user': pred_body.id_user, 'predict': pred[0]}
    data = data.append(new_row, ignore_index=True)
    data.to_csv('atm/api/predictions.csv', index=False)
    return {"prediction": pred[0]}

@app.post("/predict_items")
async def Predict_items_csv(file: UploadFile):
    content = await file.read()
    df = read_csv(StringIO(content.decode("utf-8")), sep = ',')
    X_test = preprocess(df)
    pred = model.predict(X_test)
    df['pred'] = pred
    return df[["lat","long","pred"]].to_dict(orient="records")

@app.get("/history/{id}")
def Save_feedback(id : int):
    data = read_csv('atm/api/predictions.csv')
    return data[data["id_user"] == id].to_dict(orient="records")

@app.post("/feedback")
def Save_feedback(feedback_body: Feedback):
    data = read_csv('atm/api/feedback.csv')
    new_row = {'id_user': feedback_body.id_user, 'feedback': feedback_body.feedback}
    data = data.append(new_row, ignore_index=True)
    data.to_csv('atm/api/feedback.csv', index=False)
    return "Спасибо за отзыв"

