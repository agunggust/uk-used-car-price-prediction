"""Checks for training-only comparable medians and overlap rules."""

import pandas as pd

from src.robustness import _similarity_keys, comparable_medians


def test_comparable_medians_use_most_specific_supported_group():
    train = pd.DataFrame(
        [("a", "one", 2020)] * 10
        + [("a", "one", 2010)] * 5
        + [("a", "two", 2010)] * 15
        + [("b", "three", 2020)] * 40,
        columns=["brand", "model", "year"],
    )
    prices = pd.Series([100] * 10 + [50] * 5 + [20] * 15 + [500] * 40)
    holdout = pd.DataFrame(
        [("a", "one", 2020), ("a", "one", 2010), ("a", "new", 2020), ("c", "new", 2020)],
        columns=train.columns,
        index=[7, 3, 2, 1],
    )

    result = comparable_medians(train, prices, holdout)

    assert result.index.equals(holdout.index)
    assert result["prediction"].tolist() == [100, 100, 35, 500]
    assert result["level"].tolist() == ["brand/model/age_band", "brand/model", "brand", "global"]


def test_similarity_uses_mileage_bins_and_ignores_price(listings):
    train = listings.iloc[[0]].assign(mileage=12_000)
    same_bin = train.assign(mileage=12_999, price=1)
    next_bin = train.assign(mileage=13_000)
    other_fuel = train.assign(fuelType="Electric")

    holdout = pd.concat([same_bin, next_bin, other_fuel])
    assert _similarity_keys(holdout).isin(_similarity_keys(train)).tolist() == [True, False, False]
