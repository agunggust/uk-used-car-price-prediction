"""Tests for feature importance and error analysis."""

import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.analysis import (
    aggregated_importances,
    error_by_age,
    error_by_price_band,
    error_by_segment,
    feature_importances,
    holdout_predictions,
    permutation_importances,
    run_analysis,
)
from src.train import TARGET, build_model
from xgboost import XGBRegressor


@pytest.fixture
def fitted_model(listings: pd.DataFrame):
    """Fit a small XGBoost model, matching the estimator the analysis expects."""
    estimator = build_model()
    pipeline = estimator.regressor
    pipeline.steps = [
        *pipeline.steps[:-1],
        (
            "model",
            XGBRegressor(
                n_estimators=30,
                max_depth=5,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=1,
                random_state=0,
                verbosity=0,
            ),
        )
    ]
    return estimator.fit(listings.drop(columns=TARGET), listings[TARGET])


@pytest.fixture
def data_path(tmp_path, listings: pd.DataFrame):
    path = tmp_path / "cars.csv"
    listings.to_csv(path, index=False)
    return path


@pytest.fixture
def model_path(tmp_path, fitted_model):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, path)
    return path


def test_feature_importances_cover_every_encoded_column(fitted_model) -> None:
    preprocessor = fitted_model.regressor_.named_steps["features"].named_steps[
        "preprocess"
    ]
    expected = len(preprocessor.get_feature_names_out())

    importances = feature_importances(fitted_model)

    assert set(importances.columns) == {"feature", "importance", "group", "name"}
    assert len(importances) == expected
    assert importances["importance"].between(0, 1).all()
    assert importances["importance"].gt(0).any()


def test_feature_importances_are_sorted_descending(fitted_model) -> None:
    importances = feature_importances(fitted_model)

    assert importances["importance"].is_monotonic_decreasing


def test_aggregated_importances_sum_to_one(fitted_model) -> None:
    totals = aggregated_importances(feature_importances(fitted_model))

    assert totals["importance"].sum() == pytest.approx(1.0)
    assert totals["share"].sum() == pytest.approx(1.0)
    assert totals["importance"].is_monotonic_decreasing
    assert "car_age" not in totals["name"].tolist()


def test_aggregated_importances_collapse_one_hot_groups(fitted_model) -> None:
    totals = aggregated_importances(feature_importances(fitted_model))

    names = set(totals["name"])
    assert {"brand", "model", "transmission", "fuelType"} <= names
    assert not any(name.startswith("brand_") for name in names)


def test_holdout_predictions_columns_and_identity(fitted_model, data_path) -> None:
    predictions = holdout_predictions(fitted_model, data_path)

    assert {
        "actual", "predicted", "residual", "abs_error", "pct_error", "car_age"
    } <= set(predictions.columns)
    assert predictions["abs_error"].ge(0).all()
    pd.testing.assert_series_equal(
        predictions["residual"],
        predictions["predicted"] - predictions["actual"],
        check_names=False,
    )
    assert predictions["car_age"].eq(2020 - predictions["year"]).all()


def test_error_by_segment_reports_every_brand(
    fitted_model, data_path, listings
) -> None:
    predictions = holdout_predictions(fitted_model, data_path)
    segments = error_by_segment(predictions)

    assert set(segments) == {"brand", "transmission", "fuelType"}
    assert set(segments["brand"]["brand"]) == set(listings["brand"])
    assert segments["brand"]["listings"].sum() == len(predictions)


def test_error_by_price_band_has_four_bands(fitted_model, data_path) -> None:
    predictions = holdout_predictions(fitted_model, data_path)
    bands = error_by_price_band(predictions)

    assert len(bands) == 4
    assert bands["listings"].sum() == len(predictions)


def test_error_by_age_covers_every_row(fitted_model, data_path) -> None:
    predictions = holdout_predictions(fitted_model, data_path)
    ages = error_by_age(predictions)

    assert len(ages) > 0
    assert ages["listings"].sum() == len(predictions)


def test_permutation_importance_detects_real_signal(fitted_model, listings) -> None:
    X = listings.drop(columns=TARGET)
    y = listings[TARGET]

    permuted = permutation_importances(model=fitted_model, X=X, y=y, n_repeats=3)

    assert list(permuted.columns) == ["feature", "mae_increase", "std"]
    assert set(permuted["feature"]) == set(X.columns)
    assert permuted["mae_increase"].notna().all()

    ranked = dict(zip(permuted["feature"], permuted["mae_increase"]))
    # `tax` is constant in the fixture, so shuffling it must not hurt.
    assert ranked["tax"] == pytest.approx(0.0, abs=1e-6)
    # `year` and `mileage` drive the synthetic price formula.
    assert ranked["year"] > 0
    assert ranked["mileage"] > 0


def test_run_analysis_writes_reports(
    tmp_path, data_path, model_path, listings
) -> None:
    reports_dir = tmp_path / "reports"

    results = run_analysis(
        data_path=data_path, model_path=model_path, reports_dir=reports_dir
    )

    expected = [
        "feature_importance_top.csv",
        "feature_importance_by_source.csv",
        "feature_importance_permutation.csv",
        "error_by_brand.csv",
        "error_by_transmission.csv",
        "error_by_fuelType.csv",
        "error_by_price_band.csv",
        "error_by_age.csv",
        "worst_predictions.csv",
        "analysis_summary.json",
    ]
    for name in expected:
        assert (reports_dir / name).is_file(), name

    summary = json.loads((reports_dir / "analysis_summary.json").read_text("utf-8"))
    assert summary["holdout_rows"] == len(results["predictions"])
    assert summary["holdout_metrics"]["MAE (£)"] > 0
    assert 0 <= summary["share_within_25pct"] <= 1
    assert summary["figures"]["importance"].endswith(
        "figures/analysis_feature_importance.png"
    )
    assert 0 < len(summary["top_permutation_features"]) <= 10
    assert (reports_dir / "figures" / "analysis_importance_comparison.png").is_file()

    worst = pd.read_csv(reports_dir / "worst_predictions.csv")
    assert len(worst) == min(20, len(results["predictions"]))
    assert (worst["actual"] - worst["predicted"]).abs().is_monotonic_decreasing


def test_run_analysis_reports_missing_model(tmp_path, data_path) -> None:
    with pytest.raises(FileNotFoundError, match="Model not found"):
        run_analysis(
            data_path=data_path,
            model_path=tmp_path / "missing.joblib",
            reports_dir=tmp_path / "reports",
        )


def test_feature_importances_rejects_wrong_length(fitted_model, monkeypatch) -> None:
    preprocessor = fitted_model.regressor_.named_steps["features"].named_steps[
        "preprocess"
    ]
    monkeypatch.setattr(
        preprocessor, "get_feature_names_out", lambda *a, **k: ["a", "b", "c"]
    )

    with pytest.raises(ValueError, match="encoded features"):
        feature_importances(fitted_model)


def test_feature_importances_rejects_unfitted_model() -> None:
    with pytest.raises(TypeError, match="does not expose a fitted pipeline"):
        feature_importances(LinearRegression())
