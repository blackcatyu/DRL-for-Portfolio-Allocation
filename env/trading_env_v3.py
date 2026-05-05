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
        self.last_action = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.portfolio_value = 1.0
        self.return_buffer = deque(maxlen=self.sharpe_window)
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        obs = np.concatenate([self.features[self.t], self.weights]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0)
        return obs
    
    def step(self, action):
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # log return转simple return
        log_ret = np.log(self.prices[self.t] / self.prices[self.t - 1])
        simple_ret = np.exp(log_ret) - 1
        simple_ret_with_cash = np.append(simple_ret, 0.0)
        log_ret_with_cash = np.append(log_ret, 0.0)
        
        # 漂移起点是last_action
        asset_values = self.last_action * np.exp(log_ret_with_cash)
        total = asset_values.sum()
        if total < 1e-8:
            drifted_weights = self.last_action.copy()
        else:
            drifted_weights = asset_values / total
        self.drifted_weights = drifted_weights.copy()
        
        # 交易成本基于漂移后weight和目标weight的差
        turnover = np.sum(np.abs(action - drifted_weights))
        cost = turnover * self.transaction_cost
        
        # 组合收益用action和simple return
        port_ret = np.dot(action, simple_ret_with_cash)
        
        # portfolio value更新
        self.portfolio_value *= (1 + port_ret - cost)
        
        # 存入buffer的是实际log return
        actual_return = np.log(1 + port_ret + 1e-8) - cost
        self.return_buffer.append(actual_return)
        
        # Sharpe based reward
        if len(self.return_buffer) < 2:
            reward = actual_return
        else:
            ret_array = np.array(self.return_buffer)
            mean_ret = np.mean(ret_array)
            std_ret = np.std(ret_array) + 1e-8
            reward = mean_ret / std_ret
        
        # state里存drifted_weights，漂移起点存last_action
        self.weights = drifted_weights.copy()
        self.last_action = action.copy()
        
        self.t += 1
        done = self.t >= self.T

        if done:
            obs = np.zeros(self.n_features + self.n_actions, dtype=np.float32)
        else:
            obs = self._get_obs()
            
        return obs, reward, done, False, {}
    
    def render(self):
        print(f"t={self.t}, portfolio_value={self.portfolio_value:.4f}, weights={self.weights}")