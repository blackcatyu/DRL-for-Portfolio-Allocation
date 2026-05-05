import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class MetalTradingEnvV3(gym.Env):
    def __init__(self, features, prices, transaction_cost=0.001, sharpe_window=20):
        super().__init__()
        
        self.features = features.values
        self.prices = prices.values
        self.transaction_cost = transaction_cost
        self.sharpe_window = sharpe_window
        self.n_assets = 3
        self.n_actions = 4
        self.n_features = features.shape[1]
        self.T = len(features)
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_actions,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + self.n_actions,),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        self.t = 1
        self.weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.portfolio_value = 1.0
        self.return_buffer = deque(maxlen=self.sharpe_window)
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        return np.concatenate([self.features[self.t], self.weights]).astype(np.float32)
    
    def step(self, action):
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        log_ret = np.log(self.prices[self.t] / self.prices[self.t - 1])
        log_ret_with_cash = np.append(log_ret, 0.0)
        
        # 价格变动后weight漂移
        asset_values = self.weights * np.exp(log_ret_with_cash)
        total = asset_values.sum()
        if total < 1e-8:
            drifted_weights = self.weights.copy()
        else:
            drifted_weights = asset_values / total
        self.drifted_weights = drifted_weights.copy()
        
        # 交易成本
        turnover = np.sum(np.abs(action - drifted_weights))
        cost = turnover * self.transaction_cost
        
        # portfolio return
        portfolio_return = np.dot(drifted_weights, log_ret_with_cash) - cost
        self.portfolio_value *= np.exp(portfolio_return)
        self.weights = action.copy()
        
        # 把当期return加入buffer
        self.return_buffer.append(portfolio_return)
        
        # Sharpe based reward
        if len(self.return_buffer) < 2:
            reward = portfolio_return  # buffer不够时用单步return
        else:
            ret_array = np.array(self.return_buffer)
            mean_ret = np.mean(ret_array)
            std_ret = np.std(ret_array) + 1e-8
            reward = mean_ret / std_ret  # rolling Sharpe
        
        self.t += 1
        done = self.t >= self.T

        if done:
            obs = np.zeros(self.n_features + self.n_actions, dtype=np.float32)
        else:
            obs = self._get_obs()
            
        return obs, reward, done, False, {}
    
    def render(self):
        print(f"t={self.t}, portfolio_value={self.portfolio_value:.4f}, weights={self.weights}")