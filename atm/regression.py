import pickle
from subprocess import check_output

from hydra import main
from mlflow import log_metric, log_param, set_tracking_uri, start_run
from omegaconf import DictConfig
from pandas import read_csv
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from . import data_pipeline_dir_local, model_dir_local, test_dir_local, train_dir_local
from .preprocessing.transform_data import transform_data


@main(version_base=None, config_path="./hydra", config_name="config")
def train(cfg: DictConfig):
    set_tracking_uri(f"""http://{cfg["params"].host}:{cfg["params"].port}""")
    git_commit_id = (
        check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    )
    with start_run():
        log_param("git_commit_id", git_commit_id)
        log_param("subsample", cfg["params"].subsample)
        log_param("booster", cfg["params"].booster)
        log_param("reg_lambda", cfg["params"].reg_lambda)
        log_param("max_depth", cfg["params"].max_depth)
        log_param("learning_rate", cfg["params"].learning_rate)
        log_param("n_estimators", cfg["params"].n_estimators)
        log_param("colsample_bylevel", cfg["params"].colsample_bylevel)
        log_param("colsample_bytree", cfg["params"].colsample_bytree)
        with open(data_pipeline_dir_local, "rb") as file:
            data_pipeline = pickle.load(file)
        train_data = data_pipeline.transform(
            transform_data(read_csv(train_dir_local, sep=";"))
        )
        test_data = data_pipeline.transform(
            transform_data(read_csv(test_dir_local, sep=";"))
        )
        train_target = train_data["target"]
        train_data = train_data.drop("target", axis=1)

        reg = XGBRegressor(
            subsample=cfg["params"].subsample,
            booster=cfg["params"].booster,
            reg_lambda=cfg["params"].reg_lambda,
            max_depth=cfg["params"].max_depth,
            learning_rate=cfg["params"].learning_rate,
            n_estimators=cfg["params"].n_estimators,
            colsample_bylevel=cfg["params"].colsample_bylevel,
            colsample_bytree=cfg["params"].colsample_bytree,
        ).fit(train_data, train_target)

        log_metric("r2_score", r2_score(test_data, reg.predict(test_data)))
        with open(model_dir_local, "wb") as file:
            pickle.dump(reg, file)


if __name__ == "__main__":
    train()
