"""Feature importance and holdout error analysis for the selected model."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # backend must be set before pyplot is imported

from sklearn.inspection import permutation_importance as sklearn_permutation_importance

from src.evaluate import score_regression
from src.features import add_car_age
from src.modeling import BEST_MODEL_PATH
from src.provenance import load_verified_model
from src.train import DATA_PATH, INPUT_FEATURES, load_evaluation_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
TOP_FEATURES = 20
WORST_CASES = 20
ERROR_SEGMENTS = ("brand", "transmission", "fuelType")
PERMUTATION_REPEATS = 5
RANDOM_STATE = 42


def _fitted_preprocessor(model):
    """Return the fitted ColumnTransformer inside the saved estimator."""
    if hasattr(model, "regressor_"):
        pipeline = model.regressor_
    elif hasattr(model, "regressor"):
        pipeline = model.regressor
    else:
        raise TypeError("Model does not expose a fitted preprocessing pipeline.")
    return pipeline.named_steps["features"].named_steps["preprocess"]


def _inner_regressor(model):
    if hasattr(model, "regressor_"):
        pipeline = model.regressor_
    elif hasattr(model, "regressor"):
        pipeline = model.regressor
    else:
        raise TypeError("Model does not expose a fitted pipeline.")
    return pipeline.named_steps["model"]


def feature_importances(model) -> pd.DataFrame:
    """Return encoded feature importances in descending order.

    Works for tree ensembles exposing ``feature_importances_``; the encoded
    names come from the fitted ``ColumnTransformer`` so they match the columns
    the model actually saw.
    """
    regressor = _inner_regressor(model)
    if not hasattr(regressor, "feature_importances_"):
        raise TypeError(
            f"{type(regressor).__name__} does not expose feature_importances_."
        )

    names = _fitted_preprocessor(model).get_feature_names_out()
    values = np.asarray(regressor.feature_importances_, dtype=float)
    if len(names) != len(values):
        raise ValueError(
            f"Got {len(values)} importances for {len(names)} encoded features."
        )

    frame = pd.DataFrame({"feature": names, "importance": values})
    frame["group"] = frame["feature"].str.split("__", n=1).str[0]
    frame["name"] = frame["feature"].str.split("__", n=1).str[1]
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def aggregated_importances(importances: pd.DataFrame) -> pd.DataFrame:
    """Sum encoded importances per source column.

    One-hot categories are collapsed back onto their original column, and the
    ``year`` column is reported as ``car_age`` because that is the derived
    feature the model receives.
    """
    grouped = importances.copy()
    grouped.loc[grouped["name"] == "car_age", "name"] = "year (as car_age)"
    grouped.loc[grouped["name"] == "year", "name"] = "year (as car_age)"
    for column in ("brand", "model", "transmission", "fuelType"):
        mask = grouped["name"].str.startswith(f"{column}_")
        grouped.loc[mask, "name"] = column

    totals = (
        grouped.groupby("name", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    totals["share"] = totals["importance"] / totals["importance"].sum()
    return totals


def permutation_importances(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = PERMUTATION_REPEATS
) -> pd.DataFrame:
    """Measure how much MAE worsens when each source column is shuffled.

    Unlike impurity importance this is not biased towards high-cardinality
    columns such as ``model`` (194 distinct values), so the two views can be
    compared. Values are mean MAE increase in pounds over ``n_repeats``.
    """
    result = sklearn_permutation_importance(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": X.columns,
            "mae_increase": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return frame.sort_values("mae_increase", ascending=False).reset_index(drop=True)


def holdout_predictions(model, data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Score the holdout split and return per-row error details."""
    (_, X_test, _, y_test), _ = load_evaluation_split(data_path)
    return _score_holdout(model, X_test, y_test)


def _score_holdout(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    predictions = pd.Series(model.predict(X_test), index=X_test.index)

    result = X_test.copy()
    result["actual"] = y_test
    result["predicted"] = predictions
    result["residual"] = result["predicted"] - result["actual"]
    result["abs_error"] = result["residual"].abs()
    result["pct_error"] = result["abs_error"] / result["actual"].replace(0, np.nan)
    result["car_age"] = add_car_age(result[["year"]])["car_age"].values
    return result


def error_by_segment(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarise error per category for the main grouping columns."""
    segments = {}
    for column in ERROR_SEGMENTS:
        grouped = (
            predictions.groupby(column, observed=True)
            .agg(
                listings=("abs_error", "size"),
                median_price=("actual", "median"),
                mae=("abs_error", "mean"),
                median_abs_error=("abs_error", "median"),
                bias=("residual", "mean"),
            )
            .sort_values("mae", ascending=False)
        )
        grouped["mae_pct_of_price"] = grouped["mae"] / grouped["median_price"]
        segments[column] = grouped.reset_index()
    return segments


def error_by_price_band(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise error across actual-price quartiles."""
    banded = predictions.copy()
    banded["price_band"] = pd.qcut(
        banded["actual"], q=4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 priciest"]
    )
    grouped = (
        banded.groupby("price_band", observed=True)
        .agg(
            listings=("abs_error", "size"),
            median_price=("actual", "median"),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
        )
        .reset_index()
    )
    grouped["mae_pct_of_price"] = grouped["mae"] / grouped["median_price"]
    return grouped


def error_by_age(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise error across car-age bands."""
    banded = predictions.copy()
    edges = [0, 1, 3, 5, 8, 11, 100]
    labels = ["0-1y", "2-3y", "4-5y", "6-8y", "9-11y", "12y+"]
    banded["age_band"] = pd.cut(
        banded["car_age"], bins=edges, labels=labels, include_lowest=True
    )
    grouped = (
        banded.groupby("age_band", observed=True)
        .agg(
            listings=("abs_error", "size"),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
        )
        .reset_index()
    )
    return grouped


def _plot_importance_comparison(
    by_source: pd.DataFrame, permuted: pd.DataFrame, figures_dir: Path
) -> Path:
    """Show impurity and permutation importance side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    impurity = by_source.iloc[::-1].copy()
    impurity["name"] = impurity["name"].str.replace(
        "year (as car_age)", "year", regex=False
    )
    axes[0].barh(impurity["name"], impurity["share"], color="#2a6f97")
    axes[0].set_title("Impurity importance (share)")
    axes[0].set_xlabel("Share of total importance")
    axes[0].xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    permutation = permuted.iloc[::-1]
    axes[1].barh(permutation["feature"], permutation["mae_increase"], color="#b23a48")
    axes[1].set_title("Permutation importance (MAE increase)")
    axes[1].set_xlabel("MAE increase when column is shuffled (£)")

    figure.suptitle("The two importance views disagree", fontsize=13)
    figure.tight_layout()
    path = figures_dir / "analysis_importance_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_importances(by_source: pd.DataFrame, figures_dir: Path) -> Path:
    top = by_source.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top["name"], top["share"], color="#2a6f97")
    axis.set_xlabel("Share of total impurity importance")
    axis.set_title("Feature importance by source column")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    figure.tight_layout()
    path = figures_dir / "analysis_feature_importance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_residuals(predictions: pd.DataFrame, figures_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        predictions["actual"], predictions["predicted"],
        s=4, alpha=0.25, color="#2a6f97", linewidths=0,
    )
    limit = predictions[["actual", "predicted"]].max().max() * 1.02
    axes[0].plot([0, limit], [0, limit], color="#b23a48", linewidth=1, linestyle="--")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("Actual price (£)")
    axes[0].set_ylabel("Predicted price (£)")
    axes[0].set_title("Holdout predictions vs actuals")

    axes[1].scatter(
        predictions["predicted"], predictions["residual"],
        s=4, alpha=0.25, color="#2a6f97", linewidths=0,
    )
    axes[1].axhline(0, color="#b23a48", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Predicted price (£)")
    axes[1].set_ylabel("Residual: predicted - actual (£)")
    axes[1].set_title("Residuals by predicted price")

    figure.tight_layout()
    path = figures_dir / "analysis_residuals.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_error_by_brand(segments: dict[str, pd.DataFrame], figures_dir: Path) -> Path:
    table = segments["brand"].sort_values("mae")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(table["brand"], table["mae"], color="#2a6f97")
    axis.set_xlabel("Mean absolute error (£)")
    axis.set_title("Holdout MAE by brand")
    figure.tight_layout()
    path = figures_dir / "analysis_error_by_brand.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _relative_path(path: Path) -> str:
    """Return a project-relative path when possible, else the absolute path."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_analysis(
    data_path: Path = DATA_PATH,
    model_path: Path = BEST_MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, object]:
    """Write importance, segment, and worst-case reports; return the tables."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}; run `python -m src.modeling` first."
        )
    (_, X_test, _, y_test), evaluation_input = load_evaluation_split(data_path)
    model, model_provenance = load_verified_model(model_path, evaluation_input)
    importances = feature_importances(model)
    by_source = aggregated_importances(importances)
    predictions = _score_holdout(model, X_test, y_test)
    permuted = permutation_importances(
        model, predictions.loc[:, list(INPUT_FEATURES)], predictions["actual"]
    )
    segments = error_by_segment(predictions)
    bands = error_by_price_band(predictions)
    ages = error_by_age(predictions)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importances.head(TOP_FEATURES).to_csv(
        reports_dir / "feature_importance_top.csv", index=False
    )
    by_source.to_csv(reports_dir / "feature_importance_by_source.csv", index=False)
    permuted.to_csv(reports_dir / "feature_importance_permutation.csv", index=False)
    for column, table in segments.items():
        table.to_csv(reports_dir / f"error_by_{column}.csv", index=False)
    bands.to_csv(reports_dir / "error_by_price_band.csv", index=False)
    ages.to_csv(reports_dir / "error_by_age.csv", index=False)

    worst = predictions.nlargest(WORST_CASES, "abs_error")[
        ["brand", "model", "year", "mileage", "actual", "predicted", "residual"]
    ]
    worst.to_csv(reports_dir / "worst_predictions.csv", index=False)

    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "importance": _plot_importances(by_source, figures_dir),
        "importance_comparison": _plot_importance_comparison(
            by_source, permuted, figures_dir
        ),
        "residuals": _plot_residuals(predictions, figures_dir),
        "error_by_brand": _plot_error_by_brand(segments, figures_dir),
    }

    metrics = score_regression(predictions["actual"], predictions["predicted"])
    summary = {
        "provenance": model_provenance,
        "holdout_rows": len(predictions),
        "holdout_metrics": {name: float(value) for name, value in metrics.items()},
        "mean_residual": float(predictions["residual"].mean()),
        "median_pct_error": float(predictions["pct_error"].median()),
        "share_within_10pct": float((predictions["pct_error"] <= 0.10).mean()),
        "share_within_25pct": float((predictions["pct_error"] <= 0.25).mean()),
        "top_source_features": by_source.head(10).to_dict(orient="records"),
        "top_permutation_features": permuted.head(10).to_dict(orient="records"),
        "worst_brand_by_mae": segments["brand"].iloc[0].to_dict(),
        "best_brand_by_mae": segments["brand"].iloc[-1].to_dict(),
        "figures": {
            name: _relative_path(path) for name, path in figures.items()
        },
    }
    with (reports_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=float)
        file.write("\n")

    return {
        "importances": importances,
        "by_source": by_source,
        "permutation": permuted,
        "predictions": predictions,
        "segments": segments,
        "price_bands": bands,
        "age_bands": ages,
        "worst": worst,
        "figures": figures,
        "summary": summary,
    }


def _print_report(results: dict[str, object], reports_dir: Path) -> None:
    summary = results["summary"]
    metrics = summary["holdout_metrics"]
    print(f"Holdout rows: {summary['holdout_rows']:,}")
    print(
        "Holdout metrics: "
        f"MAE £{metrics['MAE (£)']:,.0f} | "
        f"RMSE £{metrics['RMSE (£)']:,.0f} | "
        f"R² {metrics['R²']:.4f}"
    )
    print(f"Mean residual (bias): £{summary['mean_residual']:,.0f}")
    print(
        "Accuracy: "
        f"{summary['share_within_10pct']:.1%} within ±10%, "
        f"{summary['share_within_25pct']:.1%} within ±25%"
    )

    print("\nTop source features (impurity, biased toward many categories):")
    print(results["by_source"].head(10).to_string(
        index=False,
        formatters={"importance": "{:.4f}".format, "share": "{:.1%}".format},
    ))

    print("\nPermutation importance (MAE increase in £ when shuffled):")
    print(results["permutation"].to_string(
        index=False,
        formatters={
            "mae_increase": "£{:,.0f}".format,
            "std": "£{:,.0f}".format,
        },
    ))

    print("\nError by brand (worst first):")
    print(results["segments"]["brand"].to_string(
        index=False,
        formatters={
            "mae": "£{:,.0f}".format,
            "median_price": "£{:,.0f}".format,
            "median_abs_error": "£{:,.0f}".format,
            "bias": "£{:,.0f}".format,
            "mae_pct_of_price": "{:.1%}".format,
        },
    ))

    print("\nError by price band:")
    print(results["price_bands"].to_string(
        index=False,
        formatters={
            "mae": "£{:,.0f}".format,
            "median_price": "£{:,.0f}".format,
            "bias": "£{:,.0f}".format,
            "mae_pct_of_price": "{:.1%}".format,
        },
    ))

    print(f"\nReports written to {reports_dir}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Feature importance and holdout error analysis."
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, help="Cleaned CSV path."
    )
    parser.add_argument(
        "--model", type=Path, default=BEST_MODEL_PATH, help="Saved model path."
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR, help="Report output directory."
    )
    args = parser.parse_args()

    try:
        results = run_analysis(args.data, args.model, args.reports_dir)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}")

    _print_report(results, args.reports_dir)


if __name__ == "__main__":
    main()
