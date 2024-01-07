from dvc.api import DVCFileSystem
from pandas import read_csv
from joblib import load
from . import project_dir_git, train_dir_git, test_dir_git, models_dir_git

def read_data(path):
    fs = DVCFileSystem(project_dir_git)
    with fs.open(path) as file:
        if path == train_dir_git or path == test_dir_git:
            data = read_csv(file, index_col=[0])
            data = data.drop(columns=["address", "address_rus"])
            data = data.dropna()
        elif path == models_dir_git:
            data = load(file)
        else:
            raise 
    return data