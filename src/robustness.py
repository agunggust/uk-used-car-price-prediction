"""Evaluate a comparable-listing baseline and audit the historical holdout."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluate import score_regression
from src.features import add_car_age
from src.modeling import BEST_MODEL_PATH, REPORTS_DIR
from src.provenance import load_verified_model
from src.train import DATA_PATH, load_evaluation_split


MIN_COMPARABLES = 10
COMPARABLE_LEVELS = (("brand", "model", "age_band"), ("brand", "model"), ("brand",))
SIMILARITY_COLUMNS = ("brand", "model", "year", "transmission", "fuelType", "engineSize")
MILEAGE_BIN = 1_000


def _comparable_features(listings: pd.DataFrame) -> pd.DataFrame:
    comparable = add_car_age(listings)
    comparable["age_band"] = pd.cut(
        comparable["car_age"], bins=[-1, 1, 3, 5, 8, 11, float("inf")],
        labels=["0–1", "2–3", "4–5", "6–8", "9–11", "12+"],
    )
    return comparable


def comparable_medians(X_train, y_train, X_test) -> pd.DataFrame:
    """Use the most specific training median with enough comparable listings."""
    training = _comparable_features(X_train).assign(price=y_train)
    holdout = _comparable_features(X_test)
    result = pd.DataFrame(
        {"prediction": float("nan"), "level": "global"}, index=X_test.index
    )
    for level in COMPARABLE_LEVELS:
        keys = list(level)
        stats = training.groupby(keys, observed=True)["price"].agg(["size", "median"])
        eligible = stats.loc[stats["size"] >= MIN_COMPARABLES, "median"]
        medians = holdout.join(eligible.rename("candidate"), on=keys)["candidate"]
        available = result["prediction"].isna() & medians.notna()
        result.loc[available, "prediction"] = medians.loc[available]
        result.loc[available, "level"] = "/".join(keys)
    result["prediction"] = result["prediction"].fillna(y_train.median())
    return result


def _similarity_keys(listings: pd.DataFrame) -> pd.MultiIndex:
    keys = listings.loc[:, list(SIMILARITY_COLUMNS)].assign(
        mileage_bin=listings["mileage"] // MILEAGE_BIN
    )
    return pd.MultiIndex.from_frame(keys)


def _segment_scores(results: pd.DataFrame, segment: str) -> list[dict]:
    rows = []
    for label, group in results.groupby(segment, sort=True):
        rows.append({
            "segment": label,
            "listings": len(group),
            "model_mae": float((group["model"] - group["actual"]).abs().mean()),
            "model_bias": float((group["model"] - group["actual"]).mean()),
            "comparable_mae": float((group["comparable"] - group["actual"]).abs().mean()),
            "comparable_bias": float((group["comparable"] - group["actual"]).mean()),
        })
    return rows


def run_robustness(
    data_path: Path = DATA_PATH,
    model_path: Path = BEST_MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Compare against training-only medians and quantify holdout support."""
    (X_train, X_test, y_train, y_test), evaluation_input = load_evaluation_split(data_path)
    model, provenance = load_verified_model(model_path, evaluation_input)
    baseline = comparable_medians(X_train, y_train, X_test)

    model_keys = ["brand", "model"]
    training_counts = X_train.groupby(model_keys).size().rename("training_count")
    support = X_test.join(training_counts, on=model_keys)["training_count"].fillna(0)
    support_band = pd.cut(
        support, bins=[-1, 0, MIN_COMPARABLES - 1, float("inf")],
        labels=["unseen", "rare", "common"],
    )
    similar = _similarity_keys(X_test).isin(_similarity_keys(X_train))
    results = pd.DataFrame({
        "actual": y_test,
        "model": model.predict(X_test),
        "comparable": baseline["prediction"],
        "model_support": support_band,
        "similar_training_listing": similar,
    })

    report = {
        "provenance": provenance,
        "minimum_training_comparables": MIN_COMPARABLES,
        "baseline_levels": ["/".join(level) for level in COMPARABLE_LEVELS] + ["global"],
        "baseline_level_counts": baseline["level"].value_counts().to_dict(),
        "holdout_metrics": {
            name: score_regression(y_test, results[column])
            for name, column in (("selected_model", "model"), ("comparable_median", "comparable"))
        },
        "by_model_support": _segment_scores(results, "model_support"),
        "similarity": {
            "exact_match_columns": list(SIMILARITY_COLUMNS),
            "mileage_bin_width_miles": MILEAGE_BIN,
            "segments": _segment_scores(results, "similar_training_listing"),
        },
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "robustness_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run_robustness(args.data, args.model, args.reports_dir)
    print(f"Saved robustness report to {args.reports_dir / 'robustness_summary.json'}")


if __name__ == "__main__":
    main()
