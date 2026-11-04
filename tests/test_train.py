"""Tests for the split, baseline pipeline, and evaluation helpers."""

import pandas as pd
import pytest

from src.evaluate import score_regression
from src.train import (
    TARGET,
    build_model,
    load_dataset,
    split_dataset,
)


def test_split_dataset_is_deterministic(listings: pd.DataFrame) -> None:
    first = split_dataset(listings)
    second = split_dataset(listings)

    for left, right in zip(first, second):
        if isinstance(left, pd.Series):
            pd.testing.assert_series_equal(left, right)
        else:
            pd.testing.assert_frame_equal(left, right)


def test_split_dataset_is_disjoint_and_sized(listings: pd.DataFrame) -> None:
    X_train, X_test, _, _ = split_dataset(listings, test_size=0.25)

    assert len(X_train) + len(X_test) == len(listings)
    assert len(X_test) == pytest.approx(len(listings) * 0.25, abs=1)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_dataset_keeps_target_aligned(listings: pd.DataFrame) -> None:
    X_train, _, y_train, _ = split_dataset(listings)

    pd.testing.assert_series_equal(
        y_train, listings.loc[X_train.index, TARGET], check_names=False
    )


def test_build_model_fits_and_predicts_positive_prices(listings: pd.DataFrame) -> None:
    model = build_model().fit(listings.drop(columns=TARGET), listings[TARGET])

    predictions = model.predict(listings.drop(columns=TARGET))

    assert len(predictions) == len(listings)
    assert (predictions > 0).all()


def test_build_model_handles_unseen_categories(listings: pd.DataFrame) -> None:
    model = build_model().fit(listings.drop(columns=TARGET), listings[TARGET])
    unseen = listings.iloc[[0]].drop(columns=TARGET).copy()
    unseen.loc[:, "brand"] = "ferrari"
    unseen.loc[:, "model"] = "F40"

    predictions = model.predict(unseen)

    assert len(predictions) == 1
    assert predictions[0] > 0


def test_build_model_accepts_raw_input_columns(listings: pd.DataFrame) -> None:
    model = build_model().fit(listings.drop(columns=TARGET), listings[TARGET])
    preprocessor = model.regressor_.named_steps["features"].named_steps["preprocess"]

    assert set(preprocessor.feature_names_in_) == {
        "car_age", "mileage", "tax", "mpg", "engineSize",
        "brand", "model", "transmission", "fuelType",
    }
    assert "year" not in preprocessor.feature_names_in_


def test_load_dataset_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Cleaned data not found"):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_rejects_missing_columns(tmp_path, listings) -> None:
    path = tmp_path / "cars.csv"
    listings.drop(columns="fuelType").to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_load_dataset_rejects_non_positive_target(tmp_path, listings) -> None:
    frame = listings.copy()
    frame.loc[0, TARGET] = 0
    path = tmp_path / "cars.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="positive, non-missing"):
        load_dataset(path)


def test_load_dataset_reads_valid_csv(tmp_path, listings) -> None:
    path = tmp_path / "cars.csv"
    listings.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert len(loaded) == len(listings)
    assert list(loaded.columns) == list(listings.columns)


def test_score_regression_returns_expected_metrics() -> None:
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = y_true.copy()

    metrics = score_regression(y_true, y_pred)

    assert set(metrics) == {"MAE (£)", "RMSE (£)", "R²"}
    assert metrics["MAE (£)"] == pytest.approx(0.0)
    assert metrics["RMSE (£)"] == pytest.approx(0.0)
    assert metrics["R²"] == pytest.approx(1.0)


def test_score_regression_penalises_offset_predictions() -> None:
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = y_true + 50

    metrics = score_regression(y_true, y_pred)

    assert metrics["MAE (£)"] == pytest.approx(50.0)
    assert metrics["RMSE (£)"] == pytest.approx(50.0)
