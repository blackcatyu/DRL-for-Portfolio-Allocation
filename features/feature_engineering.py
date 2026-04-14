import pandas as pd
import numpy as np
import ta


def compute_log_returns(df):
    return np.log(df / df.shift(1))


def compute_rolling_std(log_returns, window=20):
    result = log_returns.rolling(window).std()
    result.columns = [f'{c}_std' for c in result.columns]
    return result


def compute_momentum(log_returns, window=20):
    result = log_returns.rolling(window).sum()
    result.columns = [f'{c}_mom' for c in result.columns]
    return result


def compute_rolling_corr(log_returns, window=20):
    return pd.DataFrame({
        'corr_gold_silver': log_returns['gold'].rolling(window).corr(log_returns['silver']),
        'corr_gold_copper': log_returns['gold'].rolling(window).corr(log_returns['copper']),
        'corr_silver_copper': log_returns['silver'].rolling(window).corr(log_returns['copper'])
    })


def compute_rsi(df, window=14):
    rsi = pd.DataFrame(index=df.index)
    for col in df.columns:
        rsi[f'{col}_rsi'] = ta.momentum.RSIIndicator(df[col], window=window).rsi()
    return rsi


def rolling_zscore(df, window=60):
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    return (df - mean) / (std + 1e-8)


def build_features(df, return_window=20, corr_window=20, rsi_window=14, zscore_window=20):
    """
    Input: df of raw prices (gold, silver, copper)
    Output: Normalised feature matrix, with rows containing NaN values removed
    """
    log_returns = compute_log_returns(df)
    rolling_std = compute_rolling_std(log_returns, return_window)
    momentum = compute_momentum(log_returns, return_window)
    rolling_corr = compute_rolling_corr(log_returns, corr_window)
    rsi = compute_rsi(df, rsi_window)

    features = pd.concat([log_returns, rolling_std, momentum, rolling_corr, rsi], axis=1)
    features = features.dropna()

    # Normalisation
    rsi_cols = ['gold_rsi', 'silver_rsi', 'copper_rsi']
    other_cols = [c for c in features.columns if c not in rsi_cols]
    features[other_cols] = rolling_zscore(features[other_cols], zscore_window)
    features[rsi_cols] = features[rsi_cols] / 100

    features = features.dropna()
    return features