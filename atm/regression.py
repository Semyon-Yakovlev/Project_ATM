from pathlib import Path
from subprocess import check_output

from catboost import CatBoostRegressor, Pool
from hydra import main
from joblib import dump
from mlflow import log_metric, log_param, set_tracking_uri, start_run
from omegaconf import DictConfig
from sklearn.metrics import r2_score

from .data import read_data


@main(version_base=None, config_path="./hydra", config_name="config")
def train(cfg: DictConfig):
    models_dir_local = Path.cwd() / "models" / "model.h5"
    X_train_dir_local = Path.cwd() / "data" / "X_train.csv"
    X_test_dir_local = Path.cwd() / "data" / "X_test.csv"
    y_train_dir_local = Path.cwd() / "data" / "y_train.csv"
    y_test_dir_local = Path.cwd() / "data" / "y_test.csv"

    set_tracking_uri(f"""http://{cfg["params"].host}:{cfg["params"].port}""")
    git_commit_id = (
        check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    )
    with start_run():
        log_param("git_commit_id", git_commit_id)
        log_param("iterations", cfg["params"].iterations)
        log_param("learning_rate", cfg["params"].learning_rate)
        log_param("loss_function", cfg["params"].loss_function)
        log_param("verbose", cfg["params"].verbose)
        log_param("min_data_in_leaf", cfg["params"].min_data_in_leaf)
        log_param("depth", cfg["params"].depth)
        log_param("l2_leaf_reg", cfg["params"].l2_leaf_reg)
        log_param("random", cfg["params"].random)
        X_test, X_train, y_train, y_test = (
            read_data(X_test_dir_local),
            read_data(X_train_dir_local),
            read_data(y_train_dir_local),
            read_data(y_test_dir_local),
        )
        val_pool = Pool(X_test, y_test)

        reg = CatBoostRegressor(
            iterations=cfg["params"].iterations,
            learning_rate=cfg["params"].learning_rate,
            loss_function=cfg["params"].loss_function,
            verbose=cfg["params"].verbose,
            min_data_in_leaf=cfg["params"].min_data_in_leaf,
            depth=cfg["params"].depth,
            l2_leaf_reg=cfg["params"].l2_leaf_reg,
            random_seed=cfg["params"].random,
            task_type=cfg["params"].task_type,
        ).fit(X_train, y_train, eval_set=val_pool)
        log_metric("r2_score", r2_score(y_test, reg.predict(X_test)))
        dump(reg, models_dir_local)


if __name__ == "__main__":
    train()
