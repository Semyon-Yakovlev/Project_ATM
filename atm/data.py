from pathlib import Path

from joblib import load
from pandas import read_csv


def read_data(path):
    data_dir_local = Path.cwd() / "data" / "train.csv"
    X_train_dir_local = Path.cwd() / "data" / "X_train.csv"
    X_test_dir_local = Path.cwd() / "data" / "X_test.csv"
    y_train_dir_local = Path.cwd() / "data" / "y_train.csv"
    y_test_dir_local = Path.cwd() / "data" / "y_test.csv"

    if path in (X_train_dir_local, X_test_dir_local):
        data = read_csv(path, index_col=[0])
    elif path in (y_train_dir_local, y_test_dir_local):
        data = read_csv(path, index_col=[0])
    elif path == data_dir_local:
        data = read_csv(path, index_col=[0])
    else:
        raise
    return data
