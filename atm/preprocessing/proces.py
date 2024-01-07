import json

import numpy as np
import pandas as pd

from .. import train_dir_local, train_dir_git
from ..data import read_data


def feature_engineering():

    train = read_data(train_dir_git)
    train.rename(columns={"Unnamed: 0": "atm_id"}, inplace=True)
    train[["lat_rad", "long_rad"]] = np.radians(train[["lat", "long"]])
    train["key"] = 0
    for category in [
        "mall",
        "bank",
        "department_store",
        "station",
        "alcohol",
        "police",
        "university",
        "railway_station",
        "aeroway_terminal",
    ]:
        lat_long = []
        with open(f"../osm_node_{category}.json", encoding="utf8") as f:
            json_data = json.load(f)
            for elem in json_data["elements"]:
                if elem["type"] == "node":
                    lat_long.append([elem["lat"], elem["lon"]])

        cat_df = pd.DataFrame(lat_long, columns=["lat", "long"])
        cat_df[["cat_lat_rad", "cat_long_rad"]] = np.radians(cat_df[["lat", "long"]])
        cat_df["key"] = 0

        cross_merge = train.merge(cat_df, on="key", how="outer")

        # Haversine distance formula
        cross_merge["lat_diff"] = cross_merge["cat_lat_rad"] - cross_merge["lat_rad"]
        cross_merge["long_diff"] = cross_merge["cat_long_rad"] - cross_merge["long_rad"]
        cross_merge["distance"] = (
            6378.137
            * 2
            * np.arcsin(
                np.sqrt(
                    np.sin(cross_merge["lat_diff"] / 2.0) ** 2
                    + np.cos(cross_merge["lat_rad"])
                    * np.cos(cross_merge["cat_lat_rad"])
                    * np.sin(cross_merge["long_diff"] / 2.0) ** 2
                )
            )
        )

        cross_merge[f"n_{category}"] = (cross_merge["distance"] < 0.3).astype(np.uint8)
        train = train.merge(
            cross_merge.groupby("atm_id")
            .aggregate({f"n_{category}": "sum"})
            .reset_index(),
            on="atm_id",
            how="left",
        )

    return train.to_csv(train_dir_local)
