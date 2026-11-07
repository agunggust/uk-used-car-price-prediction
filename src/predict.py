"""Predict used-car prices with the model saved by ``src.modeling``."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import score_regression
from src.train import INPUT_FEATURES, TARGET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "04_best_model.joblib"
PREDICTION_COLUMN = "predicted_price"
RAW_NUMERIC_FEATURES = ("year", "mileage", "tax", "mpg", "engineSize")
TEXT_FEATURES = tuple(
    feature for feature in INPUT_FEATURES if feature not in RAW_NUMERIC_FEATURES
)
SINGLE_RECORD_FIELDS = (
    "year",
    "mileage",
    "tax",
    "mpg",
    "engine_size",
    "brand",
    "car_model",
    "transmission",
    "fuel_type",
)
ARG_TO_FEATURE = {
    "engine_size": "engineSize",
    "car_model": "model",
    "fuel_type": "fuelType",
}
FEATURE_TO_FLAG = {
    "engine_size": "--engine-size",
    "car_model": "--model-name",
    "fuel_type": "--fuel-type",
}


def load_price_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Load a fitted estimator and verify it can predict."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}; run `python -m src.modeling` first."
        )
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Object at {model_path} is not a fitted estimator.")
    return model


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw listing rows before scoring.

    Returns a copy so callers' frames are not modified in place.
    """
    raw = frame.copy()
    raw.columns = raw.columns.str.strip()
    if raw.columns.has_duplicates:
        raise ValueError("Input contains duplicate column names.")

    missing = sorted(set(INPUT_FEATURES) - set(raw.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if raw.empty:
        raise ValueError("Input contains no rows to predict.")

    prepared = raw.loc[:, [*INPUT_FEATURES]].copy()
    for column in RAW_NUMERIC_FEATURES:
        if prepared[column].isna().any():
            raise ValueError(f"Column {column!r} contains missing values.")
        try:
            prepared[column] = pd.to_numeric(prepared[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column {column!r} must contain numeric values"
            ) from error
        if not np.isfinite(prepared[column]).all():
            raise ValueError(f"Column {column!r} must contain finite values.")

    for column in TEXT_FEATURES:
        prepared[column] = prepared[column].astype("string").str.strip()
        if prepared[column].isna().any() or prepared[column].eq("").any():
            raise ValueError(f"Column {column!r} contains missing or empty text.")

    year = prepared["year"]
    if not (year.between(1995, 2020) & year.mod(1).eq(0)).all():
        raise ValueError("Year must be a whole number from 1995 to 2020.")
    prepared["year"] = year.astype(int)
    for column in ("mileage", "tax"):
        if prepared[column].lt(0).any():
            raise ValueError(f"Column {column!r} must be non-negative.")
    if prepared["mpg"].le(0).any():
        raise ValueError("MPG must be positive.")
    engine_size = prepared["engineSize"]
    electric = prepared["fuelType"].eq("Electric")
    if not (engine_size.gt(0) | (engine_size.eq(0) & electric)).all():
        raise ValueError("Engine size must be positive, or zero for Electric listings.")

    return prepared


def predict_prices(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Return input rows with a ``predicted_price`` column in pounds."""
    prepared = validate_input(frame)
    result = prepared.copy()
    result[PREDICTION_COLUMN] = model.predict(prepared)
    return result


def _money(value: float) -> str:
    return f"£{value:,.0f}"


def _record_from_args(args: argparse.Namespace) -> pd.DataFrame:
    provided = [
        name for name in SINGLE_RECORD_FIELDS if getattr(args, name) is not None
    ]
    if not provided:
        raise ValueError("No input given; pass --input or the single-car fields.")
    missing = [name for name in SINGLE_RECORD_FIELDS if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "Single-car prediction needs every field; missing: "
            + ", ".join(FEATURE_TO_FLAG.get(name, f"--{name}") for name in missing)
        )
    record = {
        ARG_TO_FEATURE.get(name, name): getattr(args, name)
        for name in SINGLE_RECORD_FIELDS
    }
    return pd.DataFrame([record])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Predict UK used-car prices with the saved model."
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="Saved model path."
    )
    parser.add_argument(
        "--input", type=Path, help="CSV of listings with the training input columns."
    )
    parser.add_argument(
        "--output", type=Path, help="Optional CSV path for the predictions."
    )
    parser.add_argument("--year", help="Single-car prediction: model year.")
    parser.add_argument("--mileage", type=float, help="Single car: mileage.")
    parser.add_argument("--tax", type=float, help="Single car: annual tax (£).")
    parser.add_argument("--mpg", type=float, help="Single car: MPG.")
    parser.add_argument(
        "--engine-size", type=float, help="Single car: engine size (L)."
    )
    parser.add_argument("--brand", help="Single car: brand, e.g. audi.")
    parser.add_argument(
        "--model-name", dest="car_model", help="Single car: model, e.g. A1."
    )
    parser.add_argument("--transmission", help="Single car: Manual or Automatic.")
    parser.add_argument("--fuel-type", dest="fuel_type", help="Single car: fuel type.")
    args = parser.parse_args()

    if args.input is not None:
        if not args.input.is_file():
            parser.error(f"Input CSV not found: {args.input}")
        frame = pd.read_csv(args.input)
    else:
        try:
            frame = _record_from_args(args)
        except ValueError as error:
            parser.error(str(error))

    try:
        model = load_price_model(args.model)
        result = predict_prices(model, frame)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}")

    if args.input is None:
        print(f"Predicted price: {_money(result[PREDICTION_COLUMN].iloc[0])}")
    else:
        preview_columns = ["brand", "model", "year", "mileage", PREDICTION_COLUMN]
        print(f"Predicted {len(result):,} listings with {args.model.name}")
        print(
            result[preview_columns]
            .head(10)
            .to_string(
                index=False,
                formatters={
                    "mileage": lambda value: f"{value:,.0f}",
                    PREDICTION_COLUMN: _money,
                },
            )
        )

    if TARGET in frame.columns:
        metrics = score_regression(frame[TARGET], result[PREDICTION_COLUMN])
        print("\nActual prices found in input, batch metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:,.3f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
