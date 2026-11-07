"""Evaluate the saved Ridge baseline on the fixed holdout split."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.provenance import load_verified_model
from src.train import DATA_PATH, MODEL_PATH, load_evaluation_split


def score_regression(y_true, y_pred) -> dict[str, float]:
    """Return the baseline's MAE, RMSE, and R-squared metrics."""
    return {
        "MAE (£)": mean_absolute_error(y_true, y_pred),
        "RMSE (£)": root_mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


def evaluate_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> pd.DataFrame:
    """Compare the saved model with the median dummy on the same holdout set."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}; run `python -m src.train` first."
        )

    (X_train, X_test, y_train, y_test), evaluation_input = load_evaluation_split(data_path)
    model, _ = load_verified_model(model_path, evaluation_input)

    dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
    predictions = {
        "Median dummy": dummy.predict(X_test),
        "Ridge + log(price)": model.predict(X_test),
    }
    scores = {
        name: score_regression(y_test, prediction)
        for name, prediction in predictions.items()
    }
    return pd.DataFrame(scores).T.sort_values("MAE (£)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Ridge price baseline.")
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, help="Cleaned CSV path."
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="Saved model path."
    )
    args = parser.parse_args()

    results = evaluate_model(args.data, args.model)
    print(results.to_string(float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
