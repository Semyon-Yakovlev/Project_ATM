from pathlib import Path

project_dir_git = "https://github.com/Semyon-Yakovlev/Project_ATM/"
train_dir_git = "data/train.csv"
models_dir_git = "models/model.h5"
predict_dir_git = "data/predict.csv"
X_train_dir_git = "data/X_train.csv"
X_test_dir_git = "data/X_test.csv"
y_train_dir_git = "data/y_train.csv"
y_test_dir_git = "data/y_test.csv"

X_train_dir_local = Path.cwd() / "data" / "X_train.csv"
X_test_dir_local = Path.cwd() / "data" / "X_test.csv"
y_train_dir_local = Path.cwd() / "data" / "y_train.csv"
y_test_dir_local = Path.cwd() / "data" / "y_test.csv"
models_dir_local = Path.cwd() / "models" / "model.h5"