"""Tune candidate regressors, evaluate the selected model, and save artifacts."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from src.evaluate import score_regression
from src.train import (
    DATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    build_model,
    load_dataset,
    split_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL_PATH = MODEL_PATH
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "04_best_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
CV_SPLITS = 3
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
XGBOOST_CONFIGS = (
    {"n_estimators": 180, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 180, "max_depth": 7, "learning_rate": 0.05},
    {"n_estimators": 180, "max_depth": 5, "learning_rate": 0.1},
    {"n_estimators": 180, "max_depth": 7, "learning_rate": 0.1},
)


def build_search(cv_splits: int = CV_SPLITS) -> GridSearchCV:
    """Create a three-fold search scored by MAE on the original price scale."""
    ridge_models = [
        Ridge(alpha=alpha, solver="lsqr")
        for alpha in RIDGE_ALPHAS
    ]
    xgboost_models = [
        XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
            verbosity=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            **config,
        )
        for config in XGBOOST_CONFIGS
    ]
    return GridSearchCV(
        estimator=build_model(),
        param_grid=[
            {"regressor__model": ridge_models + xgboost_models},
        ],
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
        n_jobs=1,
        return_train_score=False,
        verbose=1,
    )


def _describe_model(estimator) -> dict:
    if hasattr(estimator, "regressor_"):
        model = estimator.regressor_.named_steps["model"]
    elif hasattr(estimator, "regressor"):
        model = estimator.regressor.named_steps["model"]
    else:
        model = estimator
    model_params = model.get_params(deep=False)
    if isinstance(model, XGBRegressor):
        keys = (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
        )
    else:
        keys = ("alpha", "solver")
    return {
        "name": type(model).__name__,
        "parameters": {key: model_params[key] for key in keys},
    }


def _make_cv_results(search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    descriptions = [
        _describe_model(params["regressor__model"])
        for params in results["params"]
    ]
    results.insert(0, "model", [item["name"] for item in descriptions])
    results.insert(1, "model_parameters", [
        json.dumps(item["parameters"], sort_keys=True) for item in descriptions
    ])
    results["mean_cv_mae"] = -results["mean_test_score"]
    results["std_cv_mae"] = results["std_test_score"]
    return results[
        ["rank_test_score", "model", "model_parameters",
         "mean_cv_mae", "std_cv_mae"]
    ].sort_values("rank_test_score")


def run_modeling(
    data_path: Path = DATA_PATH,
    baseline_model_path: Path = BASELINE_MODEL_PATH,
    model_path: Path = BEST_MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
    cv_splits: int = CV_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tune on training folds, compare on holdout, and persist the best model."""
    baseline_model_path = Path(baseline_model_path)
    if not baseline_model_path.is_file():
        raise FileNotFoundError(
            f"Baseline model not found at {baseline_model_path}; "
            "run `python -m src.train` first."
        )

    frame = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(frame)
    search = build_search(cv_splits=cv_splits)
    search.fit(X_train, y_train)

    baseline = joblib.load(baseline_model_path)
    dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
    selected = search.best_estimator_
    predictions = {
        "Median dummy": dummy.predict(X_test),
        "Ridge baseline": baseline.predict(X_test),
        "Selected model": selected.predict(X_test),
    }
    test_metrics = pd.DataFrame(
        {
            name: score_regression(y_test, prediction)
            for name, prediction in predictions.items()
        }
    ).T.sort_values("MAE (£)")

    model_path = Path(model_path)
    reports_dir = Path(reports_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, model_path)

    cv_results = _make_cv_results(search)
    cv_results.to_csv(reports_dir / "modeling_cv_results.csv", index=False)
    best_model = _describe_model(selected)
    metrics = {
        "selection_metric": (
            f"{cv_splits}-fold cross-validated MAE (£), training split only"
        ),
        "cv_splits": cv_splits,
        "best_cv_mae": float(-search.best_score_),
        "selected_model": best_model,
        "test_size": len(X_test),
        "training_size": len(X_train),
        "test_metrics": test_metrics.to_dict(orient="index"),
    }
    with (reports_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return cv_results, test_metrics, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune used-car price models and evaluate the selected model."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Cleaned CSV path.")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=BASELINE_MODEL_PATH,
        help="Ridge baseline model from step 03.",
    )
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH, help="Best model output.")
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR, help="Directory for metrics and CV results."
    )
    parser.add_argument("--cv", type=int, default=CV_SPLITS, help="Number of CV folds.")
    args = parser.parse_args()

    cv_results, test_metrics, metrics = run_modeling(
        data_path=args.data,
        baseline_model_path=args.baseline_model,
        model_path=args.model,
        reports_dir=args.reports_dir,
        cv_splits=args.cv,
    )
    print("\nCross-validation ranking (training data only):")
    print(cv_results.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSelected model: {metrics['selected_model']}")
    print(f"Best CV MAE: £{metrics['best_cv_mae']:,.2f}")
    print("\nFinal holdout comparison:")
    print(test_metrics.to_string(float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSaved model: {args.model}")
    print(f"Saved metrics: {args.reports_dir / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
