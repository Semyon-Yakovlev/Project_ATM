from .. import train_directory
from ..dataset import read_dataset


def feature_engineering(data=read_dataset()):
    train = data
    return train.to_csv(train_directory)
