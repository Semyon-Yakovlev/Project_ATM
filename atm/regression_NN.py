import pickle
from subprocess import check_output

import lightning as L
import numpy as np
import torch
from mlflow import log_metric, log_param, set_tracking_uri, start_run
from omegaconf import DictConfig
from pandas import read_csv
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hydra import main

from . import model_nn_dir_local
from .preprocessing.transform_data import transform_data


class Regression(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=22, out_features=64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )
        # Объявляется функция потерь
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

    # Настраиваются параметры обучения
    def training_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.layers(x)
        loss = self.loss_func(pred, y)
        return loss

    # Конфигурируется оптимизатор
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer


@main(version_base=None, config_path="../hydra", config_name="config")
def train(cfg: DictConfig):
    set_tracking_uri(f"""http://{cfg["params"].host}:{cfg["params"].port}""")
    git_commit_id = (
        check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    )
    with start_run():
        log_param("git_commit_id", git_commit_id)
        log_param("batch_size", 128)
        log_param("max_epochs", 30)
        with open("./atmFastapi/data/data_pipeline.pkl", "rb") as file:
            data_pipeline = pickle.load(file)
        train_data = read_csv("./data/train.csv", sep=";")
        test_data = read_csv("./data/test.csv", sep=";")

        train_target = train_data["target"]
        test_target = test_data["target"]
        train_data = data_pipeline.transform(train_data.drop("target", axis=1))
        test_data = data_pipeline.transform(
            transform_data(test_data.drop("target", axis=1))
        )
        train_dataset = TensorDataset(
            torch.from_numpy(train_data).to(torch.float32),
            torch.from_numpy(np.array(train_target)).to(torch.float32).reshape(-1, 1),
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        Pytorch_lightning_MNIST_model = Regression()
        trainer = L.Trainer(max_epochs=30)
        trainer.fit(model=Pytorch_lightning_MNIST_model, train_dataloaders=train_loader)
        predictions = trainer.predict(
            Pytorch_lightning_MNIST_model,
            dataloaders=torch.from_numpy(test_data).to(torch.float32),
        )
        log_metric("r2_score", r2_score(test_target, predictions))
        with open(model_nn_dir_local, "wb") as file:
            pickle.dump(trainer, file)


if __name__ == "__main__":
    train()
