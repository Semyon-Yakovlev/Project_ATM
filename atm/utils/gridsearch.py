from catboost import CatBoostRegressor
from hydra import main
from omegaconf import DictConfig

from ..data import read_data
from . import X_train_dir_git, y_train_dir_git


@main(version_base=None, config_path="../hydra", config_name="config")
def train(cfg: DictConfig):
    X_train, y_train = read_data(X_train_dir_git), read_data(y_train_dir_git)
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
