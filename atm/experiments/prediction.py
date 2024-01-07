from fire import Fire
from .. import models_dir_git, test_dir_git
from ..data import read_data


def predict():
    model = read_data(models_dir_git)
    test_data = read_data(test_dir_git)
    pred = model.predict(test_data)
    print(pred)
    return 

if __name__ == "__main__":
    Fire(predict)