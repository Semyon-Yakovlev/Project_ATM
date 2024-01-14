from pathlib import Path

from catboost import CatBoostRegressor
from hydra import main
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from ..data import read_data


@main(version_base=None, config_path="../hydra", config_name="config")
def train(cfg: DictConfig):
    train_dir_local = Path.cwd() / "data" / "train.csv"
    data = read_data(train_dir_local)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target"]),
        data.target,
        test_size=0.33,
        random_state=cfg["params"].random,
    )

    grid = {
        "learning_rate": [0.000001, 0.1, 0.3, 0.5, 0.9],
        "depth": [4, 6, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "min_data_in_leaf": [1, 10, 100],
    }
    model = CatBoostRegressor(
        iterations=cfg["params"].iterations, verbose=cfg["params"].verbose
    )
    model.grid_search(grid, X=X_train, y=y_train, cv=5)
    print("Random Search - Best Hyperparameters:", model.get_params())


if __name__ == "__main__":
    train()
