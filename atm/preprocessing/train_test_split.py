from hydra import main
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from ..data import read_data
from .data_manipulation import preprocess

from . import X_train_dir_local, X_test_dir_local, y_train_dir_local, y_test_dir_local, train_dir_git

@main(version_base=None, config_path="../hydra", config_name="config")
def split_data(cfg: DictConfig):

    data = read_data(train_dir_git)
    data = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target"]),
        data.target,
        test_size=cfg["params"].test_size,
        random_state=cfg["params"].random,
    )
    X_train.to_csv(X_train_dir_local)
    X_test.to_csv(X_test_dir_local)
    y_train.to_csv(y_train_dir_local)
    y_test.to_csv(y_test_dir_local)


if __name__ == "__main__":
    split_data()
