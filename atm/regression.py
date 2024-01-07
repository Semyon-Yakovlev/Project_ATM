from catboost import CatBoostRegressor
from joblib import dump
from omegaconf import DictConfig

from hydra import main

from . import models_dir_local, train_dir_git
from .data import read_data


@main(version_base=None, config_path="../hydra", config_name="config")
def train(cfg: DictConfig):
    data = read_data(train_dir_git)
    X = data.drop(columns=["target"])
    y = data.target
    reg = CatBoostRegressor(
        iterations=cfg["params"].iterations,
        learning_rate=cfg["params"].learning_rate,
        loss_function=cfg["params"].loss_function,
    ).fit(X, y)
    dump(reg, models_dir_local)


if __name__ == "__main__":
    train()
