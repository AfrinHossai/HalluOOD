# HalluOOD

Training-Free OOD and Hallucination Calibration for Vision-Language Models.

## Development setup

HalluOOD uses Python 3.11 and [uv](https://docs.astral.sh/uv/). Install `uv` using its official installation instructions, then from the repository root run:

```sh
uv python install 3.11
uv sync
```

`uv sync` creates the project environment and installs the package with its development tools.

## Tests and linting

Run the test suite with:

```sh
uv run pytest
```

Check code style with:

```sh
uv run ruff check .
```

## Project layout

- `src/halluood/` contains the Python package.
- `tests/` contains tests.
- `scripts/` contains project scripts.
- `configs/` contains configuration files.
- `docs/` contains project documentation.
