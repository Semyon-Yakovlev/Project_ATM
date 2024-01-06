from joblib import dump
from catboost import CatBoostRegressor

from . import models_directory
from .dataset import read_dataset


def train(data=read_dataset(), models_dir=models_directory):
    data = data.drop(columns=["address", "address_rus"])
    data = data.dropna()
    X = data.drop(columns=["target"])
    y = data.target
    reg = CatBoostRegressor().fit(X, y)
    return dump(reg, models_dir)
