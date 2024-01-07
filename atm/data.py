from dvc.api import DVCFileSystem
from joblib import load
from pandas import read_csv

from . import (
    X_test_dir_git,
    X_train_dir_git,
    models_dir_git,
    project_dir_git,
    y_test_dir_git,
    y_train_dir_git,
)


def read_data(path):
    fs = DVCFileSystem(project_dir_git)
    with fs.open(path) as file:
        if path in (X_train_dir_git, X_test_dir_git):
            data = read_csv(file, index_col=[0])
            data = data.drop(columns=["address", "address_rus"])
            data = data.dropna()
        elif path in (y_train_dir_git, y_test_dir_git):
            data = read_csv(file)
        elif path == models_dir_git:
            data = load(file)
        else:
            raise
    return data
