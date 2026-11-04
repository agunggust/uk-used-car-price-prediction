"""Tests for input validation and price prediction."""

import joblib
import pandas as pd
import pytest

from src.predict import (
    PREDICTION_COLUMN,
    load_price_model,
    predict_prices,
    validate_input,
)
from src.train import INPUT_FEATURES, TARGET, build_model


@pytest.fixture
def fitted_model(listings: pd.DataFrame):
    return build_model().fit(listings.drop(columns=TARGET), listings[TARGET])


def test_validate_input_selects_input_feature_order(features_frame) -> None:
    shuffled = features_frame[list(reversed(features_frame.columns))]

    prepared = validate_input(shuffled)

    assert list(prepared.columns) == list(INPUT_FEATURES)


def test_validate_input_does_not_mutate_input(features_frame) -> None:
    original = features_frame.copy()

    validate_input(features_frame)

    pd.testing.assert_frame_equal(features_frame, original)


def test_validate_input_strips_text_whitespace(features_frame) -> None:
    spaced = features_frame.copy()
    spaced["brand"] = " " + spaced["brand"] + " "

    prepared = validate_input(spaced)

    assert prepared["brand"].tolist() == features_frame["brand"].tolist()


def test_validate_input_ignores_extra_columns(features_frame) -> None:
    with_extra = features_frame.assign(notes="scraped listing")

    prepared = validate_input(with_extra)

    assert "notes" not in prepared.columns


def test_validate_input_rejects_missing_columns(features_frame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_input(features_frame.drop(columns="mpg"))


def test_validate_input_rejects_empty_frame(features_frame) -> None:
    with pytest.raises(ValueError, match="no rows"):
        validate_input(features_frame.head(0))


def test_validate_input_rejects_missing_numeric_values(features_frame) -> None:
    frame = features_frame.copy()
    frame.loc[0, "mileage"] = None

    with pytest.raises(ValueError, match="contains missing values"):
        validate_input(frame)


def test_validate_input_rejects_non_numeric_values(features_frame) -> None:
    frame = features_frame.astype({"mpg": object})
    frame.loc[0, "mpg"] = "n/a"

    with pytest.raises(ValueError, match="must contain numeric values"):
        validate_input(frame)


@pytest.mark.parametrize("bad_value", [None, ""])
def test_validate_input_rejects_missing_text(features_frame, bad_value) -> None:
    frame = features_frame.astype({"brand": object})
    frame.loc[0, "brand"] = bad_value

    with pytest.raises(ValueError, match="missing or empty text"):
        validate_input(frame)


def test_validate_input_coerces_numeric_strings(features_frame) -> None:
    frame = features_frame.astype({"year": str, "mileage": str})

    prepared = validate_input(frame)

    assert pd.api.types.is_numeric_dtype(prepared["year"])
    assert pd.api.types.is_numeric_dtype(prepared["mileage"])


def test_predict_prices_adds_prediction_column(fitted_model, features_frame) -> None:
    result = predict_prices(fitted_model, features_frame)

    assert PREDICTION_COLUMN in result.columns
    assert len(result) == len(features_frame)
    assert (result[PREDICTION_COLUMN] > 0).all()
    assert list(result.columns) == [*INPUT_FEATURES, PREDICTION_COLUMN]


def test_predict_prices_is_row_independent(fitted_model, features_frame) -> None:
    together = predict_prices(fitted_model, features_frame)
    alone = predict_prices(fitted_model, features_frame.head(1))

    assert together[PREDICTION_COLUMN].iloc[0] == pytest.approx(
        alone[PREDICTION_COLUMN].iloc[0]
    )


def test_predict_prices_keeps_input_index(fitted_model, features_frame) -> None:
    indexed = features_frame.set_index(pd.Index(range(100, 160)))

    result = predict_prices(fitted_model, indexed)

    assert result.index.tolist() == list(range(100, 160))


def test_load_price_model_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_price_model(tmp_path / "missing.joblib")


def test_load_price_model_rejects_non_estimator(tmp_path) -> None:
    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"not": "an estimator"}, path)

    with pytest.raises(TypeError, match="not a fitted estimator"):
        load_price_model(path)


def test_load_price_model_round_trips(tmp_path, fitted_model) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, path)

    loaded = load_price_model(path)

    assert hasattr(loaded, "predict")
