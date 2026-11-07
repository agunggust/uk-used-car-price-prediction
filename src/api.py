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
    load_price_model,
    validate_input,
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

        year: int | float = Field(..., description="Whole model year from 1995 to 2020.")
        mileage: float = Field(..., description="Non-negative mileage in miles.")
        tax: float = Field(..., description="Non-negative annual vehicle excise duty.")
        mpg: float = Field(..., description="Positive combined MPG.")
        engineSize: float = Field(..., description="Engine size in litres. Zero is allowed for Electric.")
        brand: str = Field(..., description="e.g. audi.")
        model: str = Field(..., description="e.g. A1.")
        transmission: str = Field(..., description="Manual or Automatic.")
        fuelType: str = Field(..., description="Petrol, Diesel, Hybrid, Electric...")

    class PredictionResponse(BaseModel):
        predicted_price: float
        input: CarInput

    class BatchResponse(BaseModel):
        predictions: list[float]
        count: int

    def _predict(cars: list[CarInput]) -> list[float]:
        frame = pd.DataFrame([car.model_dump() for car in cars])
        try:
            prepared = validate_input(frame)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            predictions = _model().predict(prepared)
        except (FileNotFoundError, TypeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return [float(value) for value in predictions]

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
        price, = _predict([car])
        return PredictionResponse(predicted_price=price, input=car)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(cars: list[CarInput]) -> BatchResponse:
        """Score many cars in one request, preserving input order."""
        if not cars:
            raise HTTPException(status_code=422, detail="Batch must not be empty.")
        return BatchResponse(predictions=_predict(cars), count=len(cars))

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
