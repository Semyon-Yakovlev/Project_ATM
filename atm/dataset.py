from pandas import read_csv

from . import train_directory


def read_dataset(dir=train_directory):
    dataset = read_csv(dir, index_col=[0])
    return dataset
