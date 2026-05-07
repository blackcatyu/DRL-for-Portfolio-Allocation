### Deep Reinforcement Learning for Portfolio Allocation in Non-Ferrous Metal Futures
#### Muyu Deng | STATS 402: Interdisciplinary Data Analysis


# 1. DATASET 


Data source: Tonghuashun (同花顺, https://www.10jqka.com.cn/)

The dataset consists of daily settlement prices for gold, silver, and copper
futures traded on the Shanghai Futures Exchange (SHFE), covering January 2013
to December 2025 (3,157 trading days, no missing values).

The data/ directory is not tracked in this repository. You must obtain and
place the data file manually before running any code.


# 2. ENVIRONMENT SETUP


This project uses Anaconda for environment management. To reproduce the
environment exactly:

    conda env create -f environment.yaml
    conda activate <env_name>

The environment.yaml file is included in the root directory of this repository
and contains all package versions used in this project.


# 3. PROJECT STRUCTURE

```plaintext
project/
├── data/                        # Raw price data (not tracked, see Section 1)
├── env/
│   ├── trading_env.py           # V1 trading environment
│   ├── trading_env_v2.py        # V2 trading environment
│   └── trading_env_v3.py        # V3 trading environment
├── features/
│   └── feature_engineering.py  # Feature construction and normalization
├── backtest/
│   └── backtest.py              # Backtesting pipeline
├── agents/                      # Trained model files
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Exploratory data analysis
│   ├── 02_train.ipynb              # PPO V1 training (V1)
│   ├── 03_train_SAC.ipynb          # SAC V1 training (v1)
│   ├── 04_train_A2C.ipynb          # A2C V1 training (V1)
│   ├── 05_train_sharpe.ipynb       # PPO and A2C training (V2)
│   ├── backtests.ipynb             # backtests
│   ├── 07_train_restep.ipynb       # PPO and A2C V2 training (V3)
│   └── results.ipynb            # Final results and visualization
├── environment.yaml             # Conda environment specification
└── README.txt                   # This file


# 4. FILE DESCRIPTIONS

--- env/trading_env.py ---
V1 trading environment (initial reward design).
Implements a Gymnasium-compatible environment for multi-asset portfolio
allocation across gold, silver, and copper futures plus cash (4 assets).

Key design:
- State space (19-dim): 15 normalized market features + 4 portfolio weights
- Action space (4-dim continuous): target portfolio weights, clipped to [0,1]
  and normalized to sum to 1, no short-selling
- Reward: drifted_weights · log_return - transaction_cost
- Known limitation: reward is computed using drifted weights from the previous
  step rather than the current action, weakening the action-reward coupling and
  causing training instability across algorithms

--- env/trading_env_v2.py ---
V2 trading environment (corrected log-return reward).
Resolves both the reward design flaw and the data leakage issue present in V1.

Key design:
- State space (19-dim): features[t-1] strictly used (no same-day leakage)
- Action space (4-dim continuous): same as V1
- Reward: log(1 + action · simple_return) - transaction_cost
- Reward is directly tied to the current action, providing unambiguous
  gradient signal for policy learning
- This is the primary environment used for final reported results

--- env/trading_env_v3.py ---
V3 trading environment (rolling Sharpe ratio reward).
Identical to V2 in state and action design; differs only in reward formulation.

Key design:
- Reward: rolling Sharpe ratio computed over a 20-step window, mu(B)/sigma(B)
- Intended to encourage risk-adjusted behavior rather than pure return
  maximization
- Empirically underperforms V2 due to non-stationarity in the reward signal

--- features/feature_engineering.py ---
Constructs and normalizes the full feature matrix from raw price data.

Functions:
- compute_log_returns(df): computes log returns for each asset
- compute_rolling_std(log_returns, window): rolling standard deviation
- compute_momentum(log_returns, window): cumulative log return over window
- compute_rolling_corr(log_returns, window): three pairwise rolling
  correlations between gold, silver, and copper
- compute_rsi(df, window): Relative Strength Index using the ta library
- rolling_zscore(df, window): rolling z-score normalization using past
  data only, avoids look-ahead bias
- build_features(df): master function that calls all of the above,
  concatenates features, applies normalization, and drops NaN rows.
  Returns a clean feature matrix ready for use in the trading environment.

Input:  pandas DataFrame with columns [gold, silver, copper] of raw prices
Output: normalized feature matrix (15 columns) with NaN rows removed

--- backtest/backtest.py ---
Runs out-of-sample backtesting for a trained agent and computes performance
metrics.

Key functionality:
- Loads a trained Stable Baselines3 model from the agents/ directory
- Runs the model on test period data (2024-2025) using the specified
  environment version
- Computes performance metrics: cumulative return, annualized return,
  Sharpe ratio, and maximum drawdown
- Supports equal-weight and buy-and-hold baseline strategies for comparison
- Returns a results dictionary and portfolio value time series for plotting


# 5. IMPORTANT NOTES


Data leakage:
    An earlier version of the environment included same-day features f_t in
    the state observation at decision step t. This constitutes data leakage,
    as same-day closing prices are not available before executing a trade.
    The corrected environments (V2 and V3) strictly use f_{t-1}. Results from
    any model trained on the leaking environment (including early runs in
    06_train_sharpe.ipynb) are not valid and should not be used for evaluation.

Agents directory:
    please view the name of agent align with training part in notbooks
    - SAC (1M steps):  approximately 2-4 hours

================================================================================
