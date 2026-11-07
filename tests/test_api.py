"""Tests for the FastAPI prediction service."""

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import build_parser, create_app
from src.predict import PREDICTION_COLUMN, main as predict_cli, predict_prices
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


@pytest.mark.parametrize("changes", [
    {"year": 2035},
    {"year": 2019.5},
    {"mileage": -5},
    {"tax": -1},
    {"mpg": 0},
    {"mileage": "Infinity"},
    {"year": "NaN"},
    {"engineSize": 0},
    {"brand": "  "},
])
def test_cli_and_api_reject_same_invalid_listings(client, tmp_path, monkeypatch, changes):
    car = {**VALID_CAR, **changes}
    path = tmp_path / "listings.csv"
    pd.DataFrame([car]).to_csv(path, index=False)
    monkeypatch.setattr("sys.argv", [
        "predict", "--input", str(path), "--model", str(client.app.state.model_path)
    ])

    assert client.post("/predict", json=car).status_code == 422
    assert client.post("/predict/batch", json=[VALID_CAR, car]).status_code == 422
    with pytest.raises(SystemExit, match="error:"):
        predict_cli()


def test_cli_and_api_accept_electric_and_unseen_model(client, tmp_path, monkeypatch):
    car = {**VALID_CAR, "engineSize": 0, "fuelType": " Electric ", "model": "Unseen model"}
    path = tmp_path / "listings.csv"
    output = tmp_path / "predictions.csv"
    pd.DataFrame([car]).to_csv(path, index=False)
    monkeypatch.setattr("sys.argv", [
        "predict", "--input", str(path), "--output", str(output),
        "--model", str(client.app.state.model_path),
    ])

    predict_cli()
    cli_price = pd.read_csv(output).loc[0, PREDICTION_COLUMN]
    response = client.post("/predict", json=car)
    batch = client.post("/predict/batch", json=[car])
    assert response.status_code == batch.status_code == 200
    assert response.json()["predicted_price"] == pytest.approx(cli_price)
    assert batch.json()["predictions"] == pytest.approx([cli_price])


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


def test_server_parser_defaults_to_localhost() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_server_parser_accepts_container_binding() -> None:
    args = build_parser().parse_args(["--host", "0.0.0.0", "--port", "9000"])

    assert args.host == "0.0.0.0"
    assert args.port == 9000
