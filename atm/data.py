import dvc.api
import joblib
from pandas import read_csv

from . import (
    X_test_dir_git,
    X_train_dir_git,
    models_dir_git,
    project_dir_git,
    train_dir_git,
    y_test_dir_git,
    y_train_dir_git,
)


def read_data(path):
    with dvc.api.open(path, project_dir_git) as file:
        if path in (X_train_dir_git, X_test_dir_git, y_train_dir_git, y_test_dir_git):
            data = read_csv(file, index_col=[0])
        elif path == train_dir_git:
            data = read_csv(file)
        elif path == models_dir_git:
            data = joblib.load(file)
        else:
            raise
    return data
