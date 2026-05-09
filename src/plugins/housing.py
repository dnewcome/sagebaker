"""Housing plugin: regression on California housing dataset.

This is the regression counterpart to DefaultPlugin — same plugin
contract, but `task = "regression"` and the metric is R² (higher is
better, bounded above by 1).

Input: any CSV/parquet with a continuous `target` column. Defaults
match data/california.csv (8 numeric features, target = median house
value in $100K units).
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .base import TrainingPlugin

# Same skip list as DefaultPlugin so Feast bookkeeping cols are dropped.
_SKIP = {"target", "signal_id", "event_timestamp"}


class HousingPlugin(TrainingPlugin):
    name = "housing"
    task = "regression"

    def prepare(self, df: pd.DataFrame):
        # 1. Extract target FIRST
        y = df["target"].astype(float)

        # 2. Build feature frame
        X = df.drop(columns=_SKIP, errors="ignore").copy()

        # 3. Feature engineering

        # Per-household ratios
        X["rooms_per_household"] = X["AveRooms"] / (X["AveOccup"] + 1e-6)
        X["bedrooms_per_room"] = X["AveBedrms"] / (X["AveRooms"] + 1e-6)
        X["population_per_household"] = X["Population"] / (X["AveOccup"] + 1e-6)
        X["bedrooms_per_household"] = X["AveBedrms"] / (X["AveOccup"] + 1e-6)
        X["rooms_per_person"] = X["AveRooms"] / (X["Population"] + 1e-6)

        # Log transforms for skewed features
        X["log_medinc"] = np.log1p(X["MedInc"])
        X["log_population"] = np.log1p(X["Population"])
        X["log_households"] = np.log1p(X["AveOccup"])
        X["log_rooms"] = np.log1p(X["AveRooms"])
        X["log_bedrooms"] = np.log1p(X["AveBedrms"])
        X["log_rooms_per_household"] = np.log1p(X["rooms_per_household"])
        X["log_population_per_household"] = np.log1p(X["population_per_household"])

        # Geographic features
        X["lat_lon"] = X["Latitude"] * X["Longitude"]
        X["lat2"] = X["Latitude"] ** 2
        X["lon2"] = X["Longitude"] ** 2
        X["geo_dist"] = np.sqrt((X["Latitude"] - 35.0) ** 2 + (X["Longitude"] + 119.0) ** 2)

        # Distance to major CA cities
        # Los Angeles: 34.05, -118.24
        X["dist_la"] = np.sqrt((X["Latitude"] - 34.05) ** 2 + (X["Longitude"] + 118.24) ** 2)
        # San Francisco: 37.77, -122.42
        X["dist_sf"] = np.sqrt((X["Latitude"] - 37.77) ** 2 + (X["Longitude"] + 122.42) ** 2)
        # San Diego: 32.72, -117.15
        X["dist_sd"] = np.sqrt((X["Latitude"] - 32.72) ** 2 + (X["Longitude"] + 117.15) ** 2)
        # San Jose: 37.34, -121.89
        X["dist_sj"] = np.sqrt((X["Latitude"] - 37.34) ** 2 + (X["Longitude"] + 121.89) ** 2)
        # Sacramento: 38.58, -121.49
        X["dist_sac"] = np.sqrt((X["Latitude"] - 38.58) ** 2 + (X["Longitude"] + 121.49) ** 2)
        # Santa Barbara: 34.42, -119.70
        X["dist_sb"] = np.sqrt((X["Latitude"] - 34.42) ** 2 + (X["Longitude"] + 119.70) ** 2)
        # Fresno: 36.74, -119.77
        X["dist_fresno"] = np.sqrt((X["Latitude"] - 36.74) ** 2 + (X["Longitude"] + 119.77) ** 2)

        # MedInc polynomial (captures nonlinear income effect)
        X["medinc2"] = X["MedInc"] ** 2
        X["medinc3"] = X["MedInc"] ** 3
        X["sqrt_medinc"] = np.sqrt(X["MedInc"].clip(lower=0))
        X["log_medinc2"] = X["log_medinc"] ** 2

        # Income x geo interactions
        X["medinc_x_lat"] = X["MedInc"] * X["Latitude"]
        X["medinc_x_lon"] = X["MedInc"] * X["Longitude"]
        X["log_medinc_x_lat"] = X["log_medinc"] * X["Latitude"]
        X["log_medinc_x_lon"] = X["log_medinc"] * X["Longitude"]

        # Income x distance interactions
        X["medinc_x_dist_la"] = X["MedInc"] * X["dist_la"]
        X["medinc_x_dist_sf"] = X["MedInc"] * X["dist_sf"]
        X["medinc_x_dist_sd"] = X["MedInc"] * X["dist_sd"]
        X["medinc_x_dist_sj"] = X["MedInc"] * X["dist_sj"]

        # House age interactions
        X["age_x_rooms"] = X["HouseAge"] * X["AveRooms"]
        X["age_x_income"] = X["HouseAge"] * X["MedInc"]
        X["age_x_rooms_per_hh"] = X["HouseAge"] * X["rooms_per_household"]
        X["age_x_lat"] = X["HouseAge"] * X["Latitude"]

        # Geo cluster features using KMeans with more clusters
        geo_coords = X[["Latitude", "Longitude"]].values
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
        X["geo_cluster"] = kmeans.fit_predict(geo_coords)
        # Distance to nearest cluster center
        cluster_centers = kmeans.cluster_centers_
        assigned = X["geo_cluster"].values
        X["dist_to_cluster_center"] = np.sqrt(
            (X["Latitude"].values - cluster_centers[assigned, 0]) ** 2 +
            (X["Longitude"].values - cluster_centers[assigned, 1]) ** 2
        )

        # Occupancy ratio
        X["occ_ratio"] = X["AveOccup"] / (X["AveRooms"] + 1e-6)

        # Min distance to any major city
        city_dists = np.column_stack([
            X["dist_la"], X["dist_sf"], X["dist_sd"], X["dist_sj"],
            X["dist_sac"], X["dist_sb"], X["dist_fresno"]
        ])
        X["min_city_dist"] = city_dists.min(axis=1)

        # Income per room ratio
        X["income_per_room"] = X["MedInc"] / (X["AveRooms"] + 1e-6)
        X["income_per_bedroom"] = X["MedInc"] / (X["AveBedrms"] + 1e-6)

        # Population density proxy
        X["pop_density"] = X["Population"] / (X["rooms_per_household"] + 1e-6)

        return X, y

    def evaluate(self, y_true, y_pred):
        return "validation_r2", float(r2_score(y_true, y_pred))

    def build_model(self, params: dict):
        return HistGradientBoostingRegressor(
            max_iter=int(params.get("max_iter", 1500)),
            max_depth=int(params.get("max_depth", 7)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            min_samples_leaf=int(params.get("min_samples_leaf", 10)),
            l2_regularization=float(params.get("l2_regularization", 0.05)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 127)),
            max_bins=int(params.get("max_bins", 255)),
            early_stopping=False,
            random_state=42,
        )

    def extra_config(self, model, X: pd.DataFrame) -> dict:
        return {}

    def prepare_data(self, output_dir: str, seed: int = 42, extra_args=None):
        """Fetch California housing from sklearn and write data/california.csv.

        Called by the root-level prepare.py dispatcher
        (``python prepare.py --plugin housing``). sklearn-bundled, no
        download required.
        """
        import hashlib
        import json
        import os
        import shutil
        from datetime import datetime, timezone

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        bunch = fetch_california_housing(as_frame=True)
        df = bunch.frame.rename(columns={"MedHouseVal": "target"})
        out_csv = os.path.join(output_dir, "california.csv")
        df.to_csv(out_csv, index=False)

        data_hash = hashlib.sha256(open(out_csv, "rb").read()).hexdigest()
        lineage = {
            "source": "sklearn.datasets.fetch_california_housing",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": data_hash,
            "dataset_n_rows": len(df),
            "feature_names": list(df.columns.drop("target")),
            "target_stats": {
                "min": float(df["target"].min()),
                "max": float(df["target"].max()),
                "mean": float(df["target"].mean()),
            },
        }
        with open(os.path.join(output_dir, "lineage.json"), "w") as f:
            json.dump(lineage, f, indent=2)

        print(f"wrote {out_csv} ({len(df)} rows × {len(df.columns)} cols)")
        print(f"wrote {output_dir}/lineage.json (sha256: {data_hash[:16]}...)")