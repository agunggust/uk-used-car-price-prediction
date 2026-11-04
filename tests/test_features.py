"""Tests for feature transformations."""

import pandas as pd
import pytest

from src.features import REFERENCE_YEAR, add_car_age


def test_add_car_age_derives_age_and_drops_year(listings: pd.DataFrame) -> None:
    result = add_car_age(listings.drop(columns="price"))

    assert "car_age" in result.columns
    assert "year" not in result.columns
    expected = REFERENCE_YEAR - listings["year"]
    pd.testing.assert_series_equal(
        result["car_age"], expected, check_names=False
    )


def test_add_car_age_does_not_mutate_input(listings: pd.DataFrame) -> None:
    frame = listings.drop(columns="price")
    original = frame.copy()

    add_car_age(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_add_car_age_is_reversible() -> None:
    frame = pd.DataFrame({"year": [1995, 2020]})

    result = add_car_age(frame)

    assert result["car_age"].tolist() == [REFERENCE_YEAR - 1995, 0]


@pytest.mark.parametrize("year", [1995, 2005, 2020])
def test_add_car_age_is_non_negative_for_known_years(year: int) -> None:
    result = add_car_age(pd.DataFrame({"year": [year]}))

    assert result["car_age"].iloc[0] >= 0
