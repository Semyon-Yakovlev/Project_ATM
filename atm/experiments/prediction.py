from fire import Fire
from .. import models_dir_git, test_dir_git, test_dir_local
from ..data import read_data


def predict():
    model = read_data(models_dir_git)
    pred = model.predict(read_data(test_dir_git))
    return pred.to_csv(test_dir_local)

if __name__ == "__main__":
    Fire(predict)