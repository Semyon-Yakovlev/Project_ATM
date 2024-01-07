from catboost import CatBoostRegressor, Pool
from joblib import dump
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from hydra import main

from . import models_dir_local, train_dir_git
from .data import read_data


@main(version_base=None, config_path="../hydra", config_name="config")
def train(cfg: DictConfig):
    data = read_data(train_dir_git)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target"]),
        data.target,
        test_size=0.33,
        random_state=cfg["params"].random,
    )
    val_pool = Pool(X_test, y_test)
    reg = CatBoostRegressor(
        iterations=cfg["params"].iterations,
        learning_rate=cfg["params"].learning_rate,
        loss_function=cfg["params"].loss_function,
        verbose=cfg["params"].verbose,
        random_seed=cfg["params"].random,
        task_type=cfg["params"].task_type,
    ).fit(
        X_train, y_train, eval_set=val_pool, use_best_model=cfg["params"].use_best_model
    )
    dump(reg, models_dir_local)


if __name__ == "__main__":
    train()
