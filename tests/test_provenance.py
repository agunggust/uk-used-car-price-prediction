"""Regression tests for evaluation against mismatched training inputs."""

import joblib
import pytest

from src import train
from src.analysis import run_analysis
from src.evaluate import evaluate_model
from src.modeling import run_modeling


@pytest.fixture
def trained_model(tmp_path, listings):
    data_path = tmp_path / "cars.csv"
    model_path = tmp_path / "ridge.joblib"
    listings.to_csv(data_path, index=False)
    train.train_model(data_path, model_path)
    return data_path, model_path


def test_evaluation_rejects_reordered_dataset(trained_model, listings):
    data_path, model_path = trained_model
    assert not evaluate_model(data_path, model_path).empty
    listings.iloc[::-1].to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="Dataset SHA-256 does not match"):
        evaluate_model(data_path, model_path)


def test_evaluation_rejects_changed_split(trained_model, monkeypatch):
    data_path, model_path = trained_model
    split = train.split_dataset
    monkeypatch.setattr(train, "split_dataset", lambda frame: split(frame, random_state=7))

    with pytest.raises(ValueError, match="Train/holdout split does not match"):
        evaluate_model(data_path, model_path)


def test_modeling_rejects_legacy_baseline_before_training(trained_model):
    data_path, model_path = trained_model
    model = joblib.load(model_path)
    del model.training_provenance_
    joblib.dump(model, model_path)

    with pytest.raises(ValueError, match="has no training provenance"):
        run_modeling(data_path, baseline_model_path=model_path)


def test_analysis_rejects_changed_dataset_before_writing(trained_model, tmp_path, listings):
    data_path, model_path = trained_model
    listings.assign(price=listings["price"] + 1).to_csv(data_path, index=False)
    reports_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="Dataset SHA-256 does not match"):
        run_analysis(data_path, model_path, reports_dir)

    assert not reports_dir.exists()
