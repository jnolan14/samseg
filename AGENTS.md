# Repository Guidelines

## Project Structure & Module Organization
- `samseg/`: main Python package (core segmentation code, CLIs under `samseg/cli/`, subregion models under `samseg/subregions/`, atlases under `samseg/atlas/`).
- `samseg/tests/`: pytest-based tests (`test_*.py`).
- `gems/`: C++/ITK-backed components used by SAMSEG.
- `ITK/`, `CMakeLists.txt`: local ITK build integration.
- `docs/`: project and migration documentation.

## Build, Test, and Development Commands
- Install in editable mode (after ITK build):
  - `ITK_DIR=ITK-install python -m pip install --editable .[test]`
- Build ITK (from repo root):
  - `mkdir ITK-build && cd ITK-build`
  - `cmake -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../ITK-install ../ITK`
  - `make install`
- Run tests:
  - `pytest samseg`
  - Example targeted run: `pytest samseg/tests/test_samseg.py -k segmentation`
- Build wheel:
  - `ITK_DIR=ITK-install python -m pip wheel . -w ./dist --no-deps`

## Coding Style & Naming Conventions
- Python style follows existing code: 4-space indentation, `snake_case` for functions/variables, `CamelCase` for classes.
- Keep modules focused by domain (e.g., subregions in `samseg/subregions/`).
- Prefer explicit, descriptive names over abbreviations.
- Match surrounding style in touched files; no project-wide autoformatter is currently enforced in repo config.

## Testing Guidelines
- Framework: `pytest` (see `setup.cfg` extras and `samseg/tests/`).
- Add tests alongside affected functionality, using `test_*.py` and `test_*` function names.
- For numerical outputs, assert tolerances explicitly (`np.testing.assert_allclose`, Dice/correlation thresholds, etc.).

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and specific (example: `Fix syntax warning in comparison`).
- Recommended format: `<area>: <what changed>` (e.g., `subregions: add thalamus channel option`).
- PRs should include:
  - clear summary of behavior changes,
  - test evidence (exact commands + results),
  - environment notes when relevant (`ITK_DIR`, `FREESURFER_HOME`, sample data assumptions).

## Configuration & Data Notes
- Subregion CLI workflows require FreeSurfer environment variables (notably `FREESURFER_HOME`; often `SUBJECTS_DIR`).
- Avoid committing large generated outputs, subject data, or temporary build artifacts.
