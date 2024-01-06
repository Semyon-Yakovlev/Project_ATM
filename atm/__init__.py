from pathlib import Path

project_directory = Path.cwd()

train_directory = project_directory / "data" / "train.csv"
models_directory = project_directory / "models" / "model.h5"
predict_directory = project_directory / "atm" / "experiments" / "predict.csv"