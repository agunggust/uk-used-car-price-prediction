"""Reusable feature transformations for the used-car price project."""

import pandas as pd


REFERENCE_YEAR = 2020


def add_car_age(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``car_age`` and remove the original model-year feature.

    The reference year matches the 2020 data collection context documented in
    the EDA notebook. The function returns a copy so callers' input frames are
    not modified in place.
    """
    transformed = frame.copy()
    transformed["car_age"] = REFERENCE_YEAR - transformed["year"]
    return transformed.drop(columns="year")
