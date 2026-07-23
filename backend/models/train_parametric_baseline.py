#!/usr/bin/env python3
"""Train the Paragon parametric Cd baseline.

RandomForest is the deterministic default because it is supported by the web
runtime. LightGBM and XGBoost remain explicit opt-in alternatives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BACKEND_ROOT / "ParametricModels" / "DrivAerNet_ParametricData.csv"
DEFAULT_OUTPUT = BACKEND_ROOT / "artifacts" / "cfa_parametric_baseline.pkl"
TARGET_COLUMN = "Average Cd"
NON_FEATURE_COLUMNS = {
    "Experiment",
    "Design",
    "Average Cd",
    "Std Cd",
    "Average Cl",
    "Std Cl",
    "Average Cl_f",
    "Std Cl_f",
    "Average Cl_r",
    "Std Cl_r",
}
CATEGORICAL_COLUMNS = ["CarRear", "Wheels"]


def add_design_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Extract useful low-cardinality categories from the Experiment ID."""
    result = data.copy()
    tokens = result["Experiment"].str.split("_", expand=True)
    result["CarRear"] = tokens[0].map(
        {"F": "Fastback", "E": "Estateback", "N": "Notchback"}
    )
    result["Wheels"] = tokens[2].map(
        {"WW": "Open detailed", "WWS": "Open smooth", "WWC": "Closed smooth"}
    )
    return result


def select_model(random_state: int, requested_model: str = "random-forest"):
    if requested_model == "random-forest":
        return "RandomForest", RandomForestRegressor(
            n_estimators=280,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    if requested_model == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM was requested but is not installed. Install web/requirements-train.txt."
            ) from exc

        return "LightGBM", LGBMRegressor(
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )

    if requested_model == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost was requested but is not installed. Install web/requirements-train.txt."
            ) from exc

        return "XGBoost", XGBRegressor(
            n_estimators=700,
            learning_rate=0.035,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )

    raise ValueError(f"Unsupported model selection: {requested_model}")


def load_xy(data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    data = add_design_categories(pd.read_csv(data_path))
    numeric_columns = [
        column
        for column in data.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not numeric_columns:
        raise ValueError("No numeric feature columns were found.")
    return data[numeric_columns + CATEGORICAL_COLUMNS], data[TARGET_COLUMN]


def train(
    data_path: Path,
    output_path: Path,
    random_state: int,
    requested_model: str = "random-forest",
) -> dict:
    x, y = load_xy(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    model_name, estimator = select_model(random_state, requested_model)
    numeric_columns = [column for column in x.columns if column not in CATEGORICAL_COLUMNS]
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric_columns),
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    model = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    artifact = {
        "model_name": model_name,
        "model": model,
        "feature_columns": list(x.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    return {
        "model_name": model_name,
        "output_path": str(output_path),
        "feature_count": len(x.columns),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Paragon parametric baseline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=("random-forest", "lightgbm", "xgboost"),
        default="random-forest",
        help="Estimator to train. RandomForest is the serving-compatible default.",
    )
    args = parser.parse_args()

    summary = train(args.data, args.output, args.random_state, args.model)
    print(summary)


if __name__ == "__main__":
    main()
