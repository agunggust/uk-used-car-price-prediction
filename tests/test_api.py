"""Tests for the FastAPI prediction service."""

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.predict import PREDICTION_COLUMN, predict_prices
from src.train import TARGET, build_model

VALID_CAR = {
    "year": 2019,
    "mileage": 20000,
    "tax": 145,
    "mpg": 55.4,
    "engineSize": 1.4,
    "brand": "audi",
    "model": "A1",
    "transmission": "Manual",
    "fuelType": "Petrol",
}


@pytest.fixture
def client(tmp_path, listings: pd.DataFrame) -> TestClient:
    model = build_model().fit(listings.drop(columns=TARGET), listings[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return TestClient(create_app(path))


def test_health_reports_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "model.joblib"}


def test_predict_returns_positive_price(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_CAR)

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_price"] > 0
    assert payload["input"]["brand"] == "audi"


def test_predict_matches_direct_model_call(client: TestClient) -> None:
    model = joblib.load(client.app.state.model_path)
    expected = predict_prices(model, pd.DataFrame([VALID_CAR]))[PREDICTION_COLUMN]

    response = client.post("/predict", json=VALID_CAR)

    assert response.json()["predicted_price"] == pytest.approx(float(expected.iloc[0]))


def test_predict_rejects_missing_fields(client: TestClient) -> None:
    response = client.post("/predict", json={"year": 2019})

    assert response.status_code == 422
    assert any("mileage" in str(item) for item in response.json()["detail"])


def test_predict_rejects_unknown_year(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID_CAR, "year": 2035})

    assert response.status_code == 422


def test_predict_rejects_extra_fields(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID_CAR, "colour": "red"})

    assert response.status_code == 422


def test_predict_rejects_negative_mileage(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID_CAR, "mileage": -5})

    assert response.status_code == 422


def test_batch_preserves_order_and_count(client: TestClient) -> None:
    cars = [
        {**VALID_CAR, "mileage": 10000},
        {**VALID_CAR, "mileage": 80000},
        {**VALID_CAR, "mileage": 40000},
    ]

    response = client.post("/predict/batch", json=cars)

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert len(payload["predictions"]) == 3
    assert all(value > 0 for value in payload["predictions"])
    assert payload["predictions"][0] != payload["predictions"][1]


def test_batch_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/predict/batch", json=[])

    assert response.status_code == 422


def test_batch_reports_invalid_entry(client: TestClient) -> None:
    response = client.post(
        "/predict/batch", json=[VALID_CAR, {**VALID_CAR, "fuelType": ""}]
    )

    assert response.status_code == 422


def test_missing_model_returns_503(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "missing.joblib"))

    response = client.post("/predict", json=VALID_CAR)

    assert response.status_code == 503
    assert "Model not found" in response.json()["detail"]


def test_health_when_model_missing(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "missing.joblib"))

    response = client.get("/health")

    assert response.json() == {"status": "ok", "model": "missing"}
