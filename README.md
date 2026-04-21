# quantumORB

`quantumORB` is an intraday futures research and backtesting codebase focused on **setup-quality prediction**, not direct price prediction.

The main research question is:

> Can intraday trading performance be improved by predicting the quality of structured trade setups, rather than predicting price movements directly?

In the current implementation, the primary prediction target is:

`P(success | setup, context)`

where:
- `success = 1` means the setup reaches take profit before stop loss
- `success = 0` means the setup reaches stop loss before take profit

The repository starts with **NQ 1-minute data**, structured setup detection, forward-safe labeling, and bar-based backtesting, then layers on baseline ML models, LSTM experiments, and reporting.

## Current Status

This repo is no longer a skeleton. It already includes:

- raw CSV ingestion for the downloaded NQ 1-minute dataset
- preprocessing into standardized parquet
- session-aware feature engineering
- multiple rule-based setup detectors
- forward-safe trade labeling
- a backtest engine with detailed trade logs and performance metrics
- setup-quality baseline models:
  - logistic regression
  - gradient boosting
  - LSTM sequence model
- walk-forward validation
- GA experimentation scaffolding
- reproducible runner scripts and report-ready output files

Latest verified test status:

- `100 passed, 1 warning`

## Repository Layout

```text
quantumORB/
  config/
    default.yaml
  data/
    raw/
    processed/
  examples/
    preprocess_nq_dataset.py
    run_setup_quality_research_pipeline.py
    build_setup_quality_sequence_dataset.py
    run_setup_quality_lstm.py
    run_setup_quality_model_comparison.py
    run_final_paper_experiment.py
    run_github_repro_pipeline.py
    ...additional strategy/debug runners
  outputs/
    final_paper_experiment/
    setup_quality_research_orb_session_vwap_retest_liquidity/
    setup_quality_lstm_orb_session_vwap_retest_liquidity/
    setup_quality_model_comparison/
    ...additional experiment folders
  src/
    backtest/
    config/
    data/
    evaluation/
    execution/
    features/
    ga/
    indicators/
    labeling/
    models/
    portfolio/
    reporting/
    setups/
    utils/
  tests/
  pyproject.toml
  requirements.txt
  run_github_repro_pipeline.bat
  run_final_paper_experiment.bat
```

## Architecture Summary

The codebase is organized as a research pipeline:

1. **Data ingestion / standardization**
   - reads downloaded intraday CSV data
   - normalizes schema
   - preserves New York time
   - outputs parquet for downstream use

2. **Feature engineering**
   - session-aware features
   - opening range features
   - volatility and momentum context
   - VWAP interaction context
   - liquidity / key-level distance features

3. **Setup detection**
   - emits candidate trade setups using rule-based logic
   - preserves entry, stop, target, side, and setup context

4. **Forward-safe labeling**
   - labels setups using only future bars
   - no lookahead in the label generation logic

5. **Backtesting**
   - bar-based execution engine
   - session-end handling
   - trade logs
   - equity curves
   - drawdown and risk metrics

6. **Modeling**
   - tabular baseline models for setup-quality prediction
   - LSTM sequence model for context windows
   - walk-forward splits for time-aware validation

7. **Reporting**
   - CSV summaries
   - JSON metrics
   - charts for paper/report use

## Main Implemented Strategy Track for the Paper

For the current paper-ready pipeline, the primary strategy track is:

- `orb_session_vwap_retest`

Current research configuration:

- instrument: `NQ`
- dataset: processed `1-minute` NQ bars
- opening range: `09:30-09:45 America/New_York`
- target mode: `liquidity`
- labeling: forward-only TP-before-SL framing
- modeling goal: predict setup success probability

This is the main strategy used by the final paper experiment runner.

## Core Modules

### Data
- [src/data](C:/Users/prana/OneDrive/Documents/quantumORB/src/data)
- handles schema normalization and preprocessing

### Features
- [src/features](C:/Users/prana/OneDrive/Documents/quantumORB/src/features)
- builds session, OR, VWAP, volatility, and contextual features

### Setups
- [src/setups](C:/Users/prana/OneDrive/Documents/quantumORB/src/setups)
- contains rule-based detectors
- important file for the paper track:
  - [src/setups/orb_session_vwap_retest.py](C:/Users/prana/OneDrive/Documents/quantumORB/src/setups/orb_session_vwap_retest.py)

### Labeling
- [src/labeling](C:/Users/prana/OneDrive/Documents/quantumORB/src/labeling)
- contains forward-safe label generation

### Models
- [src/models](C:/Users/prana/OneDrive/Documents/quantumORB/src/models)
- baseline tabular models
- sequence dataset construction
- LSTM implementation
- walk-forward split logic

### Backtest / Execution
- [src/backtest](C:/Users/prana/OneDrive/Documents/quantumORB/src/backtest)
- [src/execution](C:/Users/prana/OneDrive/Documents/quantumORB/src/execution)
- bar-based execution and trade lifecycle handling

### Evaluation / Reporting
- [src/evaluation](C:/Users/prana/OneDrive/Documents/quantumORB/src/evaluation)
- [src/reporting](C:/Users/prana/OneDrive/Documents/quantumORB/src/reporting)
- computes trading metrics and saves charts/tables

## Current Research Results

The cleanest summary comes from the final paper runner output in:

- [outputs/final_paper_experiment](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment)
- [FINAL_PAPER_RESULTS.md](C:/Users/prana/OneDrive/Documents/quantumORB/FINAL_PAPER_RESULTS.md)

### Final Comparison

Primary strategy:

- `orb_session_vwap_retest_liquidity`

Prediction target:

- `P(success | setup, context)`

Decision rule:

- take a trade only if predicted probability is greater than or equal to a threshold

Latest full-run comparison:

| model | rule | threshold | trades | win rate | profit factor | net pnl | sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_strategy | all setups | n/a | 505 | 0.6436 | 1.1021 | 1192.25 | 0.8826 |
| logistic_regression | probability filter | 0.75 | 70 | 0.7429 | 1.3283 | 397.50 | 0.9545 |
| gradient_boosting | probability filter | 0.75 | 56 | 0.6964 | 1.3521 | 312.50 | 0.9667 |
| lstm | probability filter | 0.75 | 38 | 0.5000 | 1.4681 | 3.7353 | 0.9217 |

### Interpretation

What these results mean:

- the raw strategy makes more total money because it takes many more trades
- the probability-filtered models improve trade selectivity
- logistic regression and gradient boosting both improve **profit factor** and **Sharpe** relative to the raw strategy
- the LSTM achieves the highest profit factor, but with a very small trade count and almost no total net pnl
- for a paper or class project, **logistic regression** and **gradient boosting** are the easiest practical models to defend

This supports the main research claim:

- **setup-quality prediction can improve trade selection quality even when the raw strategy itself remains unchanged**

## Reproducibility

This section is written for a TA, professor, or anyone cloning the repository onto another machine.

### 1. Requirements

- Python `3.11`
- Windows PowerShell or Command Prompt is the easiest path because the repo includes `.bat` launchers
- enough disk space for the NQ dataset and generated outputs

### 2. Install

Create a fresh environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Dataset Placement

If starting from the raw CSV, place it here:

- [data/raw/Dataset_NQ_1min_2022_2025.csv](C:/Users/prana/OneDrive/Documents/quantumORB/data/raw/Dataset_NQ_1min_2022_2025.csv)

The supported schema is:

- `timestamp ET`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `Vwap_RTH`
- `Vwap_ETH`

### 4. Preprocess the Raw CSV

```powershell
py examples/preprocess_nq_dataset.py
```

This produces:

- [data/processed/nq_1min_2022_2025.parquet](C:/Users/prana/OneDrive/Documents/quantumORB/data/processed/nq_1min_2022_2025.parquet)

### 5. Run the Final Paper Experiment

Single command:

```powershell
py examples/run_final_paper_experiment.py
```

Windows batch shortcut:

```powershell
run_final_paper_experiment.bat
```

This runs:

- tabular setup-quality research pipeline
- sequence dataset build
- LSTM evaluation
- final comparison table and figures

Outputs are saved under:

- [outputs/final_paper_experiment](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment)

### 6. Run the Broader Reproducible Research Pipeline

If you want the larger project pipeline instead of just the paper experiment:

```powershell
py examples/run_github_repro_pipeline.py
```

Windows batch shortcut:

```powershell
run_github_repro_pipeline.bat
```

## Key Output Files for Review

These are the most useful files for grading or project review:

### Final paper outputs
- [outputs/final_paper_experiment/paper_experiment_comparison.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/paper_experiment_comparison.csv)
- [outputs/final_paper_experiment/paper_experiment_summary.json](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/paper_experiment_summary.json)
- [outputs/final_paper_experiment/figures/paper_experiment_table.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_experiment_table.png)
- [outputs/final_paper_experiment/figures/paper_profit_factor.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_profit_factor.png)
- [outputs/final_paper_experiment/figures/paper_sharpe.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_sharpe.png)
- [outputs/final_paper_experiment/figures/paper_net_pnl.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_net_pnl.png)

### Intermediate research outputs
- [outputs/final_paper_experiment/baseline_tabular](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/baseline_tabular)
- [outputs/final_paper_experiment/sequence_dataset](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/sequence_dataset)
- [outputs/final_paper_experiment/lstm](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/lstm)

## Tests

Run all tests:

```powershell
py -m pytest -q
```

Latest verified result:

- `100 passed, 1 warning`

## What Is Already Implemented vs Future Work

### Already implemented
- NQ intraday data ingestion and preprocessing
- feature pipeline
- setup detection
- forward-safe labeling
- backtesting
- walk-forward validation
- baseline setup-quality modeling
- LSTM setup-quality modeling
- reporting and final comparison outputs

### Intentionally not finished yet
- full multi-asset expansion
- RL-based trading policy
- production execution/live trading
- final optimization layer as the core result
- large-scale Monte Carlo robustness suite

Those remain future-work directions rather than required parts of the current paper result.

## Recommended Claim for the Paper

The strongest defensible claim supported by the current code and results is:

> A setup-quality prediction framework can improve trade selection quality for intraday futures setups by filtering rule-based candidates using predicted success probabilities, without needing to predict price directly.

That claim is better supported than a claim that LSTM or GA is universally superior.

## Contact / Project Context

This repository was developed as an intraday futures ML research project centered on:

- structured setup detection
- forward-safe supervision
- probability-based trade filtering
- reproducible strategy evaluation

The current implementation prioritizes **clarity, reproducibility, and research defensibility** over production brokerage integration.
