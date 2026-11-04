"""Tests that the API dependency pins stay in sync with the main requirements."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUNTIME_PACKAGES = {
    "fastapi",
    "uvicorn",
    "xgboost",
    "scikit-learn",
    "pandas",
    "joblib",
}


def parse_requirements(path: Path) -> dict[str, str]:
    """Return ``{package: version}`` for pinned ``name==version`` entries."""
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = line.split(";", 1)[0].strip()
        if "==" not in requirement:
            raise AssertionError(f"{path.name}: unpinned requirement {line!r}")
        name, version = requirement.split("==", 1)
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def test_api_requirements_are_pinned() -> None:
    pins = parse_requirements(PROJECT_ROOT / "requirements-api.txt")

    assert REQUIRED_RUNTIME_PACKAGES <= set(pins)
    assert len(pins) >= 15


def test_api_pins_match_main_requirements() -> None:
    api_pins = parse_requirements(PROJECT_ROOT / "requirements-api.txt")
    main_pins = parse_requirements(PROJECT_ROOT / "requirements.txt")

    drifted = {
        name: (version, main_pins.get(name))
        for name, version in api_pins.items()
        if main_pins.get(name) != version
    }

    assert not drifted, f"requirements-api.txt drifted from requirements.txt: {drifted}"


def test_api_requirements_include_no_notebook_stack() -> None:
    api_pins = parse_requirements(PROJECT_ROOT / "requirements-api.txt")

    notebook_stack = {"jupyter", "notebook", "matplotlib", "seaborn", "pytest"}
    assert notebook_stack.isdisjoint(api_pins)
