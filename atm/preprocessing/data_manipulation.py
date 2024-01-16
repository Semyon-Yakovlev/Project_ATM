import json

import numpy as np
import pandas as pd
from fire import Fire

from ..data import read_data
from . import train_dir_git, train_dir_local


def calc_dist(lat_1, long_1, lat_2, long_2):
    lat_diff = lat_1 - lat_2
    long_diff = long_1 - long_2
    distance = (
        6378.137
        * 2
        * np.arcsin(
            np.sqrt(
                np.sin(lat_diff / 2.0) ** 2
                + np.cos(lat_1) * np.cos(lat_2) * np.sin(long_diff / 2.0) ** 2
            )
        )
    )
    return distance


def find_population(lat, long, set_data):
    distances = set_data.apply(
        lambda settlement: calc_dist(
            lat, long, settlement["latitude_rad"], settlement["longitude_rad"]
        ),
        axis=1,
    )
    return set_data.loc[distances.idxmin(), "population"]


def add_features():
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

        settlements = pd.read_csv("../data/settlements.csv", sep=";")
        settlements = settlements[settlements["population"] >= 1000]
        set_data = (
            settlements[
                ["region", "settlement", "latitude_dd", "longitude_dd", "population"]
            ]
            .groupby(by=["region", "settlement"])
            .aggregate(
                {"latitude_dd": "mean", "longitude_dd": "mean", "population": "sum"}
            )
            .reset_index()
        )
        set_data[["latitude_rad", "longitude_rad"]] = np.radians(
            set_data[["latitude_dd", "longitude_dd"]]
        )
        train["population"] = train.apply(
            lambda row: find_population(row["lat_rad"], row["long_rad"]), axis=1
        )

        density = pd.read_csv("../data/density.csv")
        density["region"] = density["region"].str.lower()
        train = train.merge(density, how="left", on="region")

        salary = pd.read_csv("../data/salary.csv")
        salary["region"] = salary["region"].str.lower()
        train = train.merge(salary, how="left", on="region")

        train.drop(
            columns=[
                "atm_id",
                "id",
                "atm_group",
                "address",
                "address_rus",
                "lat_rad",
                "long_rad",
                "key",
                "region",
            ],
            inplace=True,
        )

    return train.to_csv(train_dir_local)


if __name__ == "__main__":
    Fire(add_features)
