from fire import Fire
from numpy import savetxt

from .. import models_dir_git, predict_dir_git, X_test_dir_git
from ..data import read_data


def predict():
    model = read_data(models_dir_git)
    pred = model.predict(read_data(X_test_dir_git))
    return savetxt(predict_dir_git, pred, delimiter=",")


if __name__ == "__main__":
    Fire(predict)
