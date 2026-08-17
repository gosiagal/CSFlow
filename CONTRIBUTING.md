# Contributing

## Setup

```bash
git clone <repo-url>
cd csflow
pip install -e ".[dev]"
```

For the JiT integration, create the separate conda environment described in [jit/environment.yaml](jit/environment.yaml).

## Code style

- Format with `black .`
- Lint with `ruff check .`
- Keep functions typed and documented following the existing style in `src/`.

## Pull requests

- Keep changes focused and describe what/why in the PR description.
- Do not commit generated artifacts (`figures/`, `results/`) — these are reproducible via `main_gpu.py` / `reproduce_figures.py` and are excluded via `.gitignore`. Precomputed inputs under `rapsd_cache/` should stay tracked.
- Reference the related paper section/experiment when changing core weighting logic (`src/frequencies_utils.py`, `src/metrics_calculator.py`, `src/rapsd_calculator_gpu.py`).
