# Final Paper Results

This file is a paper-ready summary of the current `quantumORB` research results for the primary setup-quality experiment.

Search keywords:
- setup quality prediction
- intraday futures
- NQ
- opening range breakout
- VWAP retest
- walk-forward validation
- logistic regression
- gradient boosting
- LSTM
- profit factor
- Sharpe ratio
- Sortino ratio
- Calmar ratio

## Primary Experiment

Primary strategy:
- `orb_session_vwap_retest_liquidity`

Prediction target:
- `P(success | setup, context)`

Decision rule:
- take a setup only when the model probability is greater than or equal to a threshold

Dataset:
- processed NQ 1-minute dataset

Walk-forward validation:
- tabular models: `9` folds
- LSTM: `9` folds

## Main Comparison Table

Source file:
- [paper_experiment_comparison.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/paper_experiment_comparison.csv)

| model_name | selection_rule | probability_threshold | trades_executed | win_rate | profit_factor | net_pnl | max_drawdown | sharpe | sortino | calmar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_strategy | all_setups | n/a | 505 | 0.6436 | 1.1021 | 1192.2500 | -1056.7500 | 0.8826 | 1.2093 | 1.1496 |
| logistic_regression | probability_filter | 0.7500 | 70 | 0.7429 | 1.3283 | 397.5000 | -194.2500 | 0.9545 | 1.3810 | 2.0523 |
| gradient_boosting | probability_filter | 0.7500 | 56 | 0.6964 | 1.3521 | 312.5000 | -306.2500 | 0.9667 | 1.3626 | 1.0245 |
| lstm | probability_filter | 0.7500 | 38 | 0.5000 | 1.4681 | 3.7353 | -1.8613 | 0.9217 | 1.3994 | 2.0068 |

## Recommended Claims

### Claim 1: Setup-quality probability filtering improves trade selectivity

Use:
- raw strategy vs logistic regression
- raw strategy vs gradient boosting

Why:
- both filtered tabular models improved profit factor relative to the raw strategy
- both filtered tabular models improved Sharpe relative to the raw strategy
- both filtered tabular models reduced drawdown materially

### Claim 2: Gradient boosting is the strongest practical tabular model in the current experiment

Use:
- gradient boosting has the highest Sharpe among the practical filtered models: `0.9667`
- gradient boosting also has the highest practical profit factor among tabular models: `1.3521`

Why not lead with LSTM:
- LSTM has the highest profit factor overall, `1.4681`
- but it only takes `38` trades and produces almost no total net pnl, `3.7353`
- that makes it harder to defend as the best practical model

### Claim 3: Logistic regression is the simplest and most interpretable baseline

Use:
- threshold `0.75`
- `70` trades
- win rate `0.7429`
- profit factor `1.3283`
- Sharpe `0.9545`
- Calmar `2.0523`

Why:
- easiest to explain
- still improves quality metrics over the raw strategy

## Exact Figures to Cite

### Final paper summary table
- [paper_experiment_table.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_experiment_table.png)

Use this in:
- main Results section
- model comparison subsection

### Profit factor comparison
- [paper_profit_factor.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_profit_factor.png)

Use this to support:
- probability filtering improved trade quality

### Sharpe comparison
- [paper_sharpe.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_sharpe.png)

Use this to support:
- filtered models improved risk-adjusted performance

### Net PnL comparison
- [paper_net_pnl.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/figures/paper_net_pnl.png)

Use this to explain:
- the raw strategy still makes more total money because it trades much more often

## Exact Walk-Forward Files to Cite

### Tabular walk-forward validation
- [fold_results.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/baseline_tabular/fold_results.csv)
- [oos_predictions.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/baseline_tabular/oos_predictions.csv)
- [threshold_sweep.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/baseline_tabular/threshold_sweep.csv)
- [walk_forward_roc_auc.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/baseline_tabular/figures/walk_forward_roc_auc.png)
- [walk_forward_brier_score.png](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/baseline_tabular/figures/walk_forward_brier_score.png)

### LSTM walk-forward validation
- [fold_results.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/lstm/fold_results.csv)
- [oos_predictions.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/lstm/oos_predictions.csv)
- [threshold_sweep.csv](C:/Users/prana/OneDrive/Documents/quantumORB/outputs/final_paper_experiment/lstm/threshold_sweep.csv)

## Suggested Result Paragraph

Suggested wording:

> The raw ORB VWAP retest strategy generated 505 trades with a profit factor of 1.1021 and a Sharpe ratio of 0.8826. Applying probability-based setup filtering improved trade selectivity. Logistic regression at a 0.75 threshold reduced the sample to 70 trades while increasing profit factor to 1.3283, Sharpe to 0.9545, Sortino to 1.3810, and Calmar to 2.0523. Gradient boosting at the same threshold further improved Sharpe to 0.9667 and profit factor to 1.3521 across 56 trades. The LSTM model achieved the highest profit factor, 1.4681, but only generated 38 trades and produced negligible net pnl, making it less compelling as the primary practical model. Overall, these results support the claim that setup-quality prediction can improve trade selection quality without directly predicting price movement.

## Best Rows to Use in the Paper

If you want the cleanest paper presentation:

- use `raw_strategy` as the unfiltered baseline
- use `logistic_regression` as the interpretable ML baseline
- use `gradient_boosting` as the strongest practical model
- mention `lstm` as an exploratory deep-learning extension, not the headline result
