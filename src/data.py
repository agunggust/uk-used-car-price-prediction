"""Load and clean the UK used-car CSV files."""

import argparse
from pathlib import Path

import pandas as pd


BRANDS = (
    "audi",
    "bmw",
    "ford",
    "hyundi",
    "merc",
    "skoda",
    "toyota",
    "vauxhall",
    "vw",
)
REQUIRED_COLUMNS = (
    "model",
    "year",
    "price",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
    "brand",
)
NUMERIC_COLUMNS = ("year", "price", "mileage", "tax", "mpg", "engineSize")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and text values, and normalize the tax column name."""
    normalized = frame.copy()
    normalized.columns = normalized.columns.str.strip()
    tax_aliases = [
        column
        for column in normalized.columns
        if column.lower().startswith("tax(") and column.endswith(")")
    ]
    tax_columns = (["tax"] if "tax" in normalized.columns else []) + tax_aliases
    if len(tax_columns) > 1:
        raise ValueError(f"Ambiguous tax columns: {tax_columns}")
    if tax_aliases:
        normalized = normalized.rename(columns={tax_aliases[0]: "tax"})

    for column in normalized.select_dtypes(include=["object", "string"]).columns:
        normalized[column] = normalized[column].str.strip()
    return normalized


def load_raw_data(data_dir: Path = PROJECT_ROOT / "data" / "raw") -> pd.DataFrame:
    """Read and combine the nine brand CSV files used by the notebooks."""
    data_dir = Path(data_dir)
    missing_files = [
        data_dir / f"{brand}.csv"
        for brand in BRANDS
        if not (data_dir / f"{brand}.csv").is_file()
    ]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Missing required dataset files in {data_dir}: {missing}"
        )

    frames = []
    for brand in BRANDS:
        frame = normalize_columns(pd.read_csv(data_dir / f"{brand}.csv"))
        frame["brand"] = brand
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules established in ``01_data_audit_eda.ipynb``."""
    cleaned = normalize_columns(frame)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(cleaned.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    for column in NUMERIC_COLUMNS:
        try:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column {column!r} must contain numeric values"
            ) from error

    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.loc[cleaned["year"].between(1995, 2020)]
    invalid_engine = (cleaned["engineSize"] == 0) & (cleaned["fuelType"] != "Electric")
    cleaned = cleaned.loc[~invalid_engine]
    cleaned = cleaned.loc[cleaned["mpg"] >= 10].reset_index(drop=True)

    missing_values = cleaned[list(REQUIRED_COLUMNS)].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        counts = ", ".join(
            f"{column}: {count}" for column, count in missing_values.items()
        )
        raise ValueError(f"Cleaned dataset still contains missing values ({counts})")
    if cleaned.empty:
        raise ValueError("No rows remain after applying the cleaning rules")
    return cleaned


def prepare_data(
    data_dir: Path = PROJECT_ROOT / "data" / "raw",
    output_path: Path = PROJECT_ROOT / "data" / "processed" / "cars_clean.csv",
) -> pd.DataFrame:
    """Load, clean, save, and return the combined dataset."""
    cleaned = clean_data(load_raw_data(data_dir))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the used-car source CSV files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing the nine brand CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cars_clean.csv",
        help="Path for the cleaned CSV.",
    )
    args = parser.parse_args()

    cleaned = prepare_data(args.data_dir, args.output)
    print(f"Saved {len(cleaned):,} cleaned rows to {args.output}")


if __name__ == "__main__":
    main()