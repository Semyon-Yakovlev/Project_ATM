from fire import Fire
from numpy import savetxt

from .. import X_test_dir_local, models_dir_local, predict_dir_local
from ..data import read_data


def predict():
    model = read_data(models_dir_local)
    pred = model.predict(read_data(X_test_dir_local))
    return savetxt(predict_dir_local, pred, delimiter=",")


if __name__ == "__main__":
    Fire(predict)
