"""Train and save the Ridge baseline used-car price model."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import add_car_age
from src.provenance import save_model, split_index_sha256


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
    frame, _ = _read_dataset(data_path)
    return frame


def _read_dataset(data_path: Path) -> tuple[pd.DataFrame, str]:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Cleaned data not found at {data_path}; run `python -m src.data` first."
        )
    payload = data_path.read_bytes()
    frame = pd.read_csv(BytesIO(payload))
    missing = sorted(set((*INPUT_FEATURES, TARGET)) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if frame[TARGET].isna().any() or (frame[TARGET] <= 0).any():
        raise ValueError("Target 'price' must contain positive, non-missing values.")
    return frame, sha256(payload).hexdigest()


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


def load_evaluation_split(data_path: Path = DATA_PATH) -> tuple[tuple, dict]:
    """Load one CSV snapshot and identify the exact ordered train and holdout rows."""
    frame, dataset_sha256 = _read_dataset(data_path)
    split = split_dataset(frame)
    X_train, X_test, _, _ = split
    return split, {
        "dataset_sha256": dataset_sha256,
        "dataset_rows": len(frame),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(X_train),
            "holdout_rows": len(X_test),
            "train_index_sha256": split_index_sha256(X_train.index),
            "holdout_index_sha256": split_index_sha256(X_test.index),
        },
    }


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
    (X_train, _, y_train, _), evaluation_input = load_evaluation_split(data_path)
    model = build_model()
    model.fit(X_train, y_train)

    save_model(model, model_path, evaluation_input)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Ridge price baseline.")
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, help="Cleaned CSV path."
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="Model output path."
    )
    args = parser.parse_args()

    train_model(args.data, args.model)
    print(f"Saved Ridge baseline to {args.model}")


if __name__ == "__main__":
    main()
