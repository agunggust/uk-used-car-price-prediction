"""Tie saved estimators and reports to their exact evaluation inputs."""

import json
import platform
import subprocess
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def training_environment() -> dict:
    """Record the code and dependency versions used to fit an estimator."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None

    return {
        "git_head": revision,
        "source_sha256": {
            path.name: sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for path in sorted((PROJECT_ROOT / "src").glob("*.py"))
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("joblib", "numpy", "pandas", "scikit-learn", "scipy", "xgboost")
        },
    }


def save_model(model, model_path: Path, evaluation_input: dict) -> dict:
    """Save provenance inside the estimator while preserving its predict API."""
    model.training_provenance_ = {
        "evaluation_input": evaluation_input,
        "environment": training_environment(),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {
        "sha256": sha256(model_path.read_bytes()).hexdigest(),
        **model.training_provenance_,
    }


def load_verified_model(model_path: Path, evaluation_input: dict):
    """Load a model for evaluation only when dataset and partition identities match."""
    model_path = Path(model_path)
    payload = model_path.read_bytes()
    model = joblib.load(BytesIO(payload))
    provenance = getattr(model, "training_provenance_", None)
    if provenance is None:
        raise ValueError(f"Model {model_path} has no training provenance. Retrain it first.")
    expected = provenance["evaluation_input"]
    if expected["dataset_sha256"] != evaluation_input["dataset_sha256"]:
        raise ValueError(
            f"Dataset SHA-256 does not match model {model_path}. "
            "Retrain it or use its training CSV."
        )
    if expected != evaluation_input:
        raise ValueError(f"Train/holdout split does not match model {model_path}. Retrain it first.")
    return model, {"sha256": sha256(payload).hexdigest(), **provenance}


def split_index_sha256(index) -> str:
    """Hash ordered, zero-based CSV row positions independently of platform."""
    return sha256(json.dumps(index.tolist(), separators=(",", ":")).encode()).hexdigest()
