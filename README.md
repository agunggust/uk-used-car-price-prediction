# uk-used-car-price-prediction

Baseline project for predicting UK used-car listing prices. The notebooks document
the data audit, exploratory analysis, and Ridge baseline; the `src` package provides
the same preparation and training steps as repeatable command-line commands.

## Setup

Use Python 3.10 or later, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

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

## Tests

The test suite uses pytest and does not need the raw dataset or trained model
artifacts — it builds small synthetic frames and fits the pipeline on them:

```powershell
python -m pytest
```

It covers cleaning rules, feature engineering, the deterministic split, the
baseline pipeline, prediction-side input validation, and model loading.

## Notebooks

Run the notebooks in order from `notebooks/`: `01_data_audit_eda.ipynb`,
`02_eda.ipynb`, `03_preprocessing_features.ipynb`, and
`04_modeling_evaluation.ipynb`. Set the notebook working directory to
`notebooks/` as their relative paths expect. The cleaning notebook produces
the processed CSV required by the remaining notebooks.