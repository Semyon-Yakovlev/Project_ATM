from pathlib import Path

train_dir_git = "data/train.csv"

X_train_dir_local = Path.cwd() / "data" / "X_train.csv"
X_test_dir_local = Path.cwd() / "data" / "X_test.csv"
y_train_dir_local = Path.cwd() / "data" / "y_train.csv"
y_test_dir_local = Path.cwd() / "data" / "y_test.csv"
models_dir_local = Path.cwd() / "models" / "model.h5"
