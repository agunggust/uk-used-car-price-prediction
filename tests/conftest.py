"""Shared fixtures for the used-car price test suite."""

import pandas as pd
import pytest


@pytest.fixture
def listings() -> pd.DataFrame:
    """Return a small, valid raw listing frame covering every input column."""
    rows = []
    for index in range(60):
        year = 2010 + index % 11
        mileage = 2000 + index * 700
        price = 22000 - (year - 2010) * 900 - mileage // 40
        rows.append(
            {
                "model": "A1" if index % 2 else "A3",
                "year": year,
                "price": price,
                "transmission": "Manual" if index % 2 else "Automatic",
                "mileage": mileage,
                "fuelType": "Petrol" if index % 3 else "Diesel",
                "tax": 145,
                "mpg": 55.4,
                "engineSize": 1.4 if index % 2 else 2.0,
                "brand": "audi" if index % 2 else "bmw",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def features_frame(listings: pd.DataFrame) -> pd.DataFrame:
    """Return listing rows without the target, as passed to ``predict``."""
    return listings.drop(columns="price")
