from pathlib import Path

project_directory = Path.cwd()
project_dir_git = "https://github.com/Semyon-Yakovlev/Project_ATM/"

train_dir_local = project_directory / "data" / "train.csv"
test_dir_local = project_directory / "data" / "test.csv"
models_dir_local = project_directory / "models" / "model.h5"
predict_dir_local = project_directory / "data" / "predict.csv"

train_dir_git = "data/train.csv"
test_dir_git = "data/test.csv"
models_dir_git = "models/model.h5"
predict_dir_git = "data/predict.csv"
