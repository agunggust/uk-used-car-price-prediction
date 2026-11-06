"""Tests for loading, normalizing, and cleaning the raw CSV files."""

import pandas as pd
import pytest

from src.data import BRANDS, clean_data, load_raw_data, normalize_columns


def test_normalize_columns_strips_names_and_text() -> None:
    frame = pd.DataFrame(
        {
            " model ": [" A1 "],
            "tax(£)": [145],
            "brand": [" audi "],
        }
    )

    result = normalize_columns(frame)

    assert "tax" in result.columns
    assert "tax(£)" not in result.columns
    assert result.loc[0, "model"] == "A1"
    assert result.loc[0, "brand"] == "audi"


def test_normalize_columns_rejects_ambiguous_tax_columns() -> None:
    frame = pd.DataFrame({"tax": [145], "tax(£)": [145]})

    with pytest.raises(ValueError, match="Ambiguous tax columns"):
        normalize_columns(frame)


def test_normalize_columns_does_not_mutate_input() -> None:
    frame = pd.DataFrame({" model ": [" A1 "]})
    original = frame.copy()

    normalize_columns(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_clean_data_removes_invalid_rows(listings: pd.DataFrame) -> None:
    dirty = pd.concat(
        [
            listings,
            pd.DataFrame(
                [
                    {**listings.iloc[0].to_dict(), "year": 1990},
                    {**listings.iloc[0].to_dict(), "year": 2021},
                    {**listings.iloc[0].to_dict(), "engineSize": 0.0},
                    {**listings.iloc[0].to_dict(), "mpg": 5.0},
                ]
            ),
        ],
        ignore_index=True,
    )

    cleaned = clean_data(dirty)

    assert cleaned["year"].between(1995, 2020).all()
    assert (cleaned["mpg"] >= 10).all()
    non_electric = cleaned["fuelType"] != "Electric"
    assert (cleaned.loc[non_electric, "engineSize"] != 0).all()


def test_clean_data_keeps_electric_zero_engine_size(listings: pd.DataFrame) -> None:
    electric = listings.iloc[[0]].copy()
    electric.loc[:, "engineSize"] = 0.0
    electric.loc[:, "fuelType"] = "Electric"

    cleaned = clean_data(electric)

    assert len(cleaned) == 1
    assert cleaned["engineSize"].iloc[0] == 0.0


def test_clean_data_drops_duplicates(listings: pd.DataFrame) -> None:
    duplicated = pd.concat([listings, listings], ignore_index=True)

    cleaned = clean_data(duplicated)

    assert len(cleaned) == len(listings)
    assert not cleaned.duplicated().any()


def test_clean_data_normalizes_model_labels_before_deduplication(
    listings: pd.DataFrame,
) -> None:
    row = listings.iloc[0].to_dict()
    frame = pd.DataFrame(
        [{**row, "model": label} for label in ("SL CLASS", "X-CLASS", "SL Class")]
    )

    assert clean_data(frame)["model"].tolist() == ["SL Class", "X Class"]


def test_clean_data_rejects_missing_values(listings: pd.DataFrame) -> None:
    frame = listings.copy()
    frame.loc[0, "mileage"] = None

    with pytest.raises(ValueError, match="missing values"):
        clean_data(frame)


def test_clean_data_rejects_non_numeric_values(listings: pd.DataFrame) -> None:
    frame = listings.copy()
    frame["mileage"] = frame["mileage"].astype(object)
    frame.loc[0, "mileage"] = "unknown"

    with pytest.raises(ValueError, match="must contain numeric values"):
        clean_data(frame)


def test_clean_data_rejects_missing_columns(listings: pd.DataFrame) -> None:
    frame = listings.drop(columns="mpg")

    with pytest.raises(ValueError, match="missing required columns"):
        clean_data(frame)


def test_load_raw_data_combines_all_brands(tmp_path) -> None:
    for brand in BRANDS:
        pd.DataFrame(
            {
                "model": ["A1"],
                "year": [2019],
                "price": [15000],
                "transmission": ["Manual"],
                "mileage": [10000],
                "fuelType": ["Petrol"],
                "tax": [145],
                "mpg": [55.4],
                "engineSize": [1.4],
            }
        ).to_csv(tmp_path / f"{brand}.csv", index=False)

    combined = load_raw_data(tmp_path)

    assert len(combined) == len(BRANDS)
    assert set(combined["brand"]) == set(BRANDS)


def test_load_raw_data_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required dataset files"):
        load_raw_data(tmp_path)
