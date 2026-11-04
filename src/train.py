"""Train and save the Ridge baseline used-car price model."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import add_car_age


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cars_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "03_ridge_baseline.joblib"
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
NUMERIC_FEATURES = ("car_age", "mileage", "tax", "mpg", "engineSize")
CATEGORICAL_FEATURES = ("brand", "model", "transmission", "fuelType")
INPUT_FEATURES = (
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize",
    "brand",
    "model",
    "transmission",
    "fuelType",
)


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset and validate fields needed for training."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Cleaned data not found at {data_path}; run `python -m src.data` first."
        )
    frame = pd.read_csv(data_path)
    missing = sorted(set(INPUT_FEATURES + (TARGET,)) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if frame[TARGET].isna().any() or (frame[TARGET] <= 0).any():
        raise ValueError("Target 'price' must contain positive, non-missing values.")
    return frame


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Split features and target using the baseline notebook's fixed seed."""
    X = frame.drop(columns=TARGET)
    y = frame[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def build_model() -> TransformedTargetRegressor:
    """Build the notebook's preprocessing and log-target Ridge pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    feature_pipeline = Pipeline(
        steps=[
            ("add_car_age", FunctionTransformer(add_car_age, validate=False)),
            ("preprocess", preprocessor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("features", feature_pipeline),
                ("model", Ridge(alpha=10.0, solver="lsqr")),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> TransformedTargetRegressor:
    """Fit on the training split only and persist the fitted estimator."""
    frame = load_dataset(data_path)
    X_train, _, y_train, _ = split_dataset(frame)
    model = build_model()
    model.fit(X_train, y_train)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Ridge price baseline.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Cleaned CSV path.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Model output path.")
    args = parser.parse_args()

    train_model(args.data, args.model)
    print(f"Saved Ridge baseline to {args.model}")


if __name__ == "__main__":
    main()