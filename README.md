# uk-used-car-price-prediction

[![CI](https://github.com/agunggust/uk-used-car-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/agunggust/uk-used-car-price-prediction/actions/workflows/ci.yml)

Baseline project for predicting UK used-car listing prices. The notebooks document
the data audit, exploratory analysis, and Ridge baseline; the `src` package provides
the same preparation and training steps as repeatable command-line commands.

## Setup

Use Python 3.12 or later (the pinned dependencies require it), then install the
project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The same suite runs on GitHub Actions via `.github/workflows/ci.yml` for every
push and pull request to `main`; CI installs `requirements.txt` and runs
`python -m pytest`. The test suite needs neither the raw dataset nor the saved
model files.

## Dataset

Download the UK used-car CSV files and put these nine files in `data/raw/`:
`audi.csv`, `bmw.csv`, `ford.csv`, `hyundi.csv`, `merc.csv`, `skoda.csv`,
`toyota.csv`, `vauxhall.csv`, and `vw.csv`. The source dataset is not included
in this repository.

## Reproduce the baseline and model selection

Run these commands from the repository root:

```powershell
python -m src.data
python -m src.train
python -m src.evaluate
python -m src.modeling
python -m src.analysis
```

The cleaning step writes `data/processed/cars_clean.csv`. Training uses the
fixed 80/20 split (random state 42), fits preprocessing only on the training
partition, and saves `models/03_ridge_baseline.joblib`. Evaluation reports MAE,
RMSE, and R² for the saved Ridge model and a median-price dummy on the same
holdout split. The model uses `car_age` relative to 2020, scaled numeric
features, one-hot encoded categorical features, and a log-transformed price
target.

The model-selection step compares three Ridge regularization strengths and
four XGBoost configurations using three-fold cross-validation on the training
split. It selects by cross-validated MAE, then evaluates the selected estimator,
the step-03 Ridge baseline, and the median dummy on the holdout set. The test
metrics do not affect model selection. Outputs are saved to
`models/04_best_model.joblib`, `reports/modeling_cv_results.csv`, and
`reports/model_metrics.json`. Import the saved estimator and call `.predict()`
with a DataFrame containing the input columns from `cars_clean.csv` except
`price`; preprocessing is included in the estimator.

To customize file locations, use `--data-dir` and `--output` with
`python -m src.data`, or `--data` and `--model` with `src.train` and
`src.evaluate`. The model-selection command also accepts `--baseline-model`,
`--reports-dir`, and `--cv`.

## Analyze the model

`python -m src.analysis` scores the same holdout split as `src.modeling` and
writes CSV reports, a JSON summary, and four figures:

| Output | Contents |
|---|---|
| `reports/feature_importance_top.csv` | 20 most important encoded features |
| `reports/feature_importance_by_source.csv` | impurity importance summed per source column |
| `reports/feature_importance_permutation.csv` | MAE increase when each source column is shuffled |
| `reports/error_by_{brand,transmission,fuelType}.csv` | MAE, median error, and bias per category |
| `reports/error_by_price_band.csv` / `error_by_age.csv` | error across price quartiles and age bands |
| `reports/worst_predictions.csv` | the 20 largest holdout misses |
| `reports/analysis_summary.json` | headline metrics, accuracy bands, both importance views |
| `reports/figures/analysis_*.png` | importance bars, the two importance views compared, residual scatter, MAE by brand |

Impurity importance is read from the fitted XGBoost booster and the encoded
names come from the fitted `ColumnTransformer`, so one-hot categories are
collapsed back onto their source column. Because impurity importance is biased
towards columns with many split points (`model` has 194 distinct values), the
command also runs permutation importance — shuffling each source column through
the full pipeline and measuring how much MAE worsens. The command accepts
`--data`, `--model`, and `--reports-dir`.

## Findings

Numbers below come from the holdout split (19,483 listings) and can be
regenerated with `python -m src.analysis`.

**Overall accuracy.** MAE £1,319, RMSE £2,295, R² 0.948; the median listing is
off by 6.2%, 71.1% fall within ±10% and 97.7% within ±25%. Mean residual is
−£172, so the model slightly under-prices overall.

**Error grows with price.** MAE rises from £618 on the cheapest quartile to
£2,464 on the priciest, and relative error climbs from 7.7% to 9.2% with a
−£955 bias in the top quartile: expensive cars are systematically under-valued.
Reporting a single £1,319 MAE therefore flatters the cheap end, where most
listings sit.

**New cars are the hardest.** The 0–1 year band has the highest MAE (£1,890)
versus £931 for 6–8 year cars — new-car pricing is driven by trim and options
that are not in the dataset.

**Brands.** Mercedes (£2,104), BMW (£1,880) and Audi (£1,815) have the largest
absolute errors because their cars cost more, but on a relative basis they sit
near the pack (9.1–9.4%); Vauxhall is easiest at £756 (7.6%).

**The two importance views disagree.** Impurity importance ranks `model` at
51% and `transmission` at 24% while `mileage` shows 0.4%. Permutation
importance — which shuffles each source column through the whole pipeline —
tells a different story: `engineSize` (+£2,448 MAE), `year` (+£2,152),
`brand` (+£1,155) and `mileage` (+£949) matter most, while `model` adds only
£871 and `tax` £57. The impurity numbers are inflated by `model` having 194
distinct values, so use `reports/feature_importance_permutation.csv` when
deciding which data to collect next.

**Outliers are real data problems.** The worst miss is a 2011 Mercedes SL
listed at £149,948 but predicted at £21,344; three of the five worst cases are
Mercedes G-Class. Treat these as candidates for data cleaning rather than as
model failures.

**Do not read into tiny segments.** `Electric` and transmission `Other` have
two listings each in the holdout, so their reported errors are noise.

**Limitations.** The dataset is asking prices collected around 2020 and the
split is random rather than time-based, so the model is not validated on later
market conditions. There is no trim, condition, or damage history, and the
model returns a point estimate with no uncertainty band.

## Predict prices

`src.predict` scores new listings with the saved model. Pass a CSV with the same
input columns as `cars_clean.csv` minus `price`:

```powershell
python -m src.predict --input listings.csv --output predictions.csv
```

Or score a single car with flags:

```powershell
python -m src.predict --year 2019 --mileage 20000 --tax 145 --mpg 55.4 `
  --engine-size 1.4 --brand audi --model-name A1 --transmission Manual --fuel-type Petrol
```

Predictions are written to a `predicted_price` column. Missing or non-numeric
inputs raise an error instead of being silently imputed, and the command uses
`models/04_best_model.joblib` unless `--model` points elsewhere. If the input
CSV still contains `price`, the command also prints MAE, RMSE, and R² for that
batch.

## REST API

`src.api` serves the saved model over HTTP:

```powershell
python -m src.api
```

Then open `http://127.0.0.1:8000/docs` for the generated interactive docs.
Endpoints:

- `GET /health` — reports whether the model file is present.
- `POST /predict` — one car, returns `predicted_price` in pounds.
- `POST /predict/batch` — a JSON array of cars, returns predictions in input order.

```powershell
$body = @{
  year = 2019; mileage = 20000; tax = 145; mpg = 55.4; engineSize = 1.4
  brand = "audi"; model = "A1"; transmission = "Manual"; fuelType = "Petrol"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" -Body $body
```

Field constraints (year 1995–2020, non-negative mileage and tax, positive MPG
and engine size, non-empty text) are validated by Pydantic, so invalid payloads
return HTTP 422 instead of being scored. If the model file is missing the
endpoint returns HTTP 503 while `GET /health` still answers. To point the app
at a different model, call `create_app(model_path)` from `src.api` in your own
script.

## Tests

The test suite uses pytest and does not need the raw dataset or trained model
artifacts — it builds small synthetic frames and fits the pipeline on them:

```powershell
python -m pytest
```

It covers cleaning rules, feature engineering, the deterministic split, the
baseline pipeline, prediction-side input validation, model loading, feature
importance and error analysis, and the API contracts (including rejection of
invalid payloads).

## Notebooks

Run the notebooks in order from `notebooks/`: `01_data_audit_eda.ipynb`,
`02_eda.ipynb`, `03_preprocessing_features.ipynb`, and
`04_modeling_evaluation.ipynb`. Set the notebook working directory to
`notebooks/` as their relative paths expect. The cleaning notebook produces
the processed CSV required by the remaining notebooks.