from pathlib import Path

project_directory = Path.cwd()
project_dir_git = "https://github.com/Semyon-Yakovlev/Project_ATM/"

data_dir_local = project_directory / "data" / "data.csv"
models_dir_local = project_directory / "models" / "model.h5"
predict_dir_local = project_directory / "data" / "predict.csv"
X_train_dir_local = project_directory / "data" / "X_train.csv"
X_test_dir_local = project_directory / "data" / "X_test.csv"
y_train_dir_local = project_directory / "data" / "y_train.csv"
y_test_dir_local = project_directory / "data" / "y_test.csv"


data_dir_git = "data/data.csv"
models_dir_git = "models/model.h5"
predict_dir_git = "data/predict.csv"
X_train_dir_git = "data/X_train.csv"
X_test_dir_git = "data/X_test.csv"
y_train_dir_git = "data/y_train.csv"
y_test_dir_git = "data/y_test.csv"