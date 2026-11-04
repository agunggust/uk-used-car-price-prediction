"""FastAPI service exposing the saved used-car price model."""

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import (
    DEFAULT_MODEL_PATH,
    PREDICTION_COLUMN,
    load_price_model,
    predict_prices,
)


def create_app(model_path: Path = DEFAULT_MODEL_PATH) -> FastAPI:
    """Build the application; the model is loaded lazily on first use."""
    app = FastAPI(
        title="UK Used Car Price API",
        description=(
            "Predict UK used-car listing prices with the model saved by "
            "`python -m src.modeling`."
        ),
        version="1.0.0",
    )
    model_path = Path(model_path)
    app.state.model_path = model_path

    @lru_cache(maxsize=1)
    def _model():
        return load_price_model(model_path)

    class CarInput(BaseModel):
        """A single car listing. Extra fields are rejected."""

        model_config = ConfigDict(extra="forbid")

        year: int = Field(..., ge=1995, le=2020, description="Model year.")
        mileage: float = Field(..., ge=0, description="Mileage in miles.")
        tax: float = Field(..., ge=0, description="Annual vehicle excise duty.")
        mpg: float = Field(..., gt=0, description="Combined MPG.")
        engineSize: float = Field(..., gt=0, description="Engine size in litres.")
        brand: str = Field(..., min_length=1, description="e.g. audi.")
        model: str = Field(..., min_length=1, description="e.g. A1.")
        transmission: str = Field(
            ..., min_length=1, description="Manual or Automatic."
        )
        fuelType: str = Field(
            ..., min_length=1, description="Petrol, Diesel, Hybrid..."
        )

    class PredictionResponse(BaseModel):
        predicted_price: float
        input: CarInput

    class BatchResponse(BaseModel):
        predictions: list[float]
        count: int

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report whether the model file is available."""
        return {
            "status": "ok",
            "model": str(model_path.name) if model_path.is_file() else "missing",
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(car: CarInput) -> PredictionResponse:
        """Score one car and return the price in pounds."""
        frame = pd.DataFrame([car.model_dump()])
        try:
            price = predict_prices(_model(), frame)[PREDICTION_COLUMN].iloc[0]
        except (FileNotFoundError, TypeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error))
        return PredictionResponse(predicted_price=float(price), input=car)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(cars: list[CarInput]) -> BatchResponse:
        """Score many cars in one request, preserving input order."""
        if not cars:
            raise HTTPException(status_code=422, detail="Batch must not be empty.")
        frame = pd.DataFrame([car.model_dump() for car in cars])
        try:
            prices = predict_prices(_model(), frame)[PREDICTION_COLUMN]
        except (FileNotFoundError, TypeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error))
        return BatchResponse(
            predictions=[float(value) for value in prices], count=len(cars)
        )

    return app


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    """Parse the server host and port."""
    parser = argparse.ArgumentParser(description="Serve the used-car price API.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Interface to bind; use 0.0.0.0 in Docker."
    )
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind.")
    return parser


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
