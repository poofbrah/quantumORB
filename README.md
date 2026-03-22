# quantumORB

`quantumORB` is a custom-built Python research and backtesting framework for intraday futures strategy development, starting with an Opening Range Breakout (ORB) baseline and expanding later into setup-quality modeling, GA-based optimization, and RL-based refinement.

The repository is intentionally structured so the core framework can still run without Qlib or FinRL. Those remain optional integrations rather than architectural dependencies.

## Project Priorities

1. ORB baseline first
2. Setup-quality / probability model second
3. GA and RL refinement later
4. Optional Qlib and FinRL adapters only after the core framework is stable

## Phase 1 Scope

Phase 1 creates the initial framework skeleton:

- repository structure
- packaging and dependency configuration
- default config
- typed domain schemas for intraday research and backtesting
- placeholder modules for future expansion
- optional integration scaffolding for Qlib and FinRL

No strategy logic, data ingestion pipelines, or backtest engine implementation is included yet.

## Architecture Direction

The framework is organized around a few principles:

- `core custom-built`: domain models and backtest interfaces belong to this repo
- `intraday futures aware`: schemas are shaped for session-based, contract-based research
- `research-friendly`: setup generation, labeling, modeling, and optimization are separate modules
- `extensible`: ORB comes first, and the first end-to-end strategy workflow will be ORB setup detection wired into the execution and evaluation layers
- `optional integrations`: external libraries should wrap the core, not replace it

## Repository Layout

```text
project_root/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  config/
    default.yaml
  data/
    raw/
    processed/
  notebooks/
  examples/
  src/
    __init__.py
    config/
    data/
    indicators/
    features/
    setups/
    labeling/
    models/
    backtest/
    execution/
    portfolio/
    evaluation/
    reporting/
    ga/
    rl/
    integrations/
    utils/
  tests/
```

## Core Domain Objects

Phase 1 introduces typed schemas for:

- `MarketBar`: intraday futures OHLCV bars
- `SetupEvent`: candidate setup emitted by rule logic such as ORB detection
- `LabeledSetup`: supervised-learning ready setup record with outcome metadata
- `Trade`: normalized trade lifecycle representation
- `Prediction`: model output for setup scoring or ranking
- `BacktestResult`: aggregate result object for evaluations and reporting

These schemas are intended to become the stable contract across the later data, backtest, feature, labeling, and modeling layers.

## Optional Integrations

`src/integrations/qlib_adapter.py` and `src/integrations/finrl_adapter.py` are scaffolding only. They exist to make later integration points explicit without introducing hard dependencies into the base framework.

## Getting Started

Create an environment and install the base package:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run tests after installing the editable package:

```bash
python -m pytest -q
```

## Planned Sequence

1. Phase 1: skeleton, packaging, typed schemas
2. Phase 2: data layer, config loading, indicators, and feature pipeline
3. Phase 3: minimal execution/backtest components required to support the first ORB workflow
4. Phase 4: ORB baseline strategy and evaluation flow
5. Phase 5: setup-quality feature and labeling pipeline
6. Phase 6: supervised setup-quality model
7. Phase 7: GA optimization
8. Phase 8: RL refinement
