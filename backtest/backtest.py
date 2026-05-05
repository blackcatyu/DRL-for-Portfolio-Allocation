import numpy as np
import pandas as pd


def run_agent(model, env):
    """
    Run the trained agent through a complete episode in the environment
    Record the portfolio value, weights and reward at each step
    """
    obs, _ = env.reset()
    done = False
    
    portfolio_values = [1.0]
    weights_history = []
    rewards = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        portfolio_values.append(env.portfolio_value)
        weights_history.append(env.weights.copy())
        rewards.append(reward)
    
    return {
        'portfolio_values': np.array(portfolio_values),
        'weights_history': np.array(weights_history),
        'rewards': np.array(rewards)
    }


def run_equal_weight(env):
    """
    Baseline 1: Equal weighting, rebalanced daily to 25%
    """
    obs, _ = env.reset()
    done = False
    
    portfolio_values = [1.0]
    equal_action = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    
    while not done:
        obs, reward, done, _, _ = env.step(equal_action)
        portfolio_values.append(env.portfolio_value)
    
    return np.array(portfolio_values)


def run_buy_and_hold(env):
    """
    Baseline 2: Buy and hold, with equal initial weightings and no subsequent rebalancing
    """
    obs, _ = env.reset()
    done = False
    
    portfolio_values = [1.0]
    
    # Step 1: Buy on an equal-weight basis
    initial_action = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    obs, reward, done, _, _ = env.step(initial_action)
    portfolio_values.append(env.portfolio_value)
    
    # Then remain stationary; action equals the current drifted weights
    while not done:
        obs, reward, done, _, _ = env.step(env.weights)
        portfolio_values.append(env.portfolio_value)
    
    return np.array(portfolio_values)


def compute_metrics(portfolio_values, freq=252):
    """
    Calculate evaluation metrics
    portfolio_values: a sequence of daily net asset values, starting from 1.0
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Total returns
    cumulative_return = portfolio_values[-1] - 1.0
    
    # Annualised return
    n_days = len(portfolio_values) - 1
    annualized_return = (portfolio_values[-1] ** (freq / n_days)) - 1
    
    # Annualised volatility
    annualized_vol = np.std(returns) * np.sqrt(freq)
    
    # Sharpe ratio (assuming a risk-free rate of 0%)
    sharpe = annualized_return / (annualized_vol + 1e-8)
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }