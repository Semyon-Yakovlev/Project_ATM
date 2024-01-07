from joblib import dump
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from hydra import main
from .. import (
    X_test_dir_local,
    X_train_dir_local,
    data_dir_git,
    y_test_dir_local,
    y_train_dir_local
)
from ..data import read_data
@main(version_base=None, config_path="../hydra", config_name="config")
def split_data(cfg: DictConfig):
    data = read_data(data_dir_git)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target"]),
        data.target,
        test_size=cfg["params"].test_size,
        random_state=cfg["params"].random,
    )
    dump(X_train, X_train_dir_local)
    dump(X_test, X_test_dir_local)
    dump(y_train, y_train_dir_local)
    dump(y_test, y_test_dir_local)

if __name__ == "__main__":
    split_data()