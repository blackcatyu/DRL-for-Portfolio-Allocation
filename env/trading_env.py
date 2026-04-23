import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class MetalTradingEnv(gym.Env):
    def __init__(self, features, prices, transaction_cost=0.001):
        super().__init__()
        
        self.features = features.values      # shape: (T, 15)
        self.prices = prices.values          # shape: (T, 3)
        self.transaction_cost = transaction_cost
        self.n_assets = 3
        self.n_actions = 4                   # 3个资产 + 现金
        self.n_features = features.shape[1]
        self.T = len(features)
        
        # action space：3个资产+现金，共4个权重，和为1，不允许做空
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_actions,), dtype=np.float32
        )
        
        # observation space：当天的15个特征 + 当前持仓权重4个（含现金）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + self.n_actions,),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        self.t = 1
        # 初始等权重，现金占25%
        self.weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.portfolio_value = 1.0
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        obs = np.concatenate([self.features[self.t], self.weights]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0)  # 把任何NaN替换成0
        return obs
    
    def step(self, action):
        # 归一化action，确保权重和为1
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # 当天log return（现金return为0）
        log_ret = np.log(self.prices[self.t] / self.prices[self.t - 1])
        log_ret_with_cash = np.append(log_ret, 0.0)  # 现金不产生收益
        
        # 价格变动后weight自动漂移
         # 价格变动后weight自动漂移
        asset_values = self.weights * np.exp(log_ret_with_cash)
        total_value = asset_values.sum()
        
        if total_value <= 0 or np.isnan(total_value):
            drifted_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        else:
            drifted_weights = asset_values / total_value
        
        # 交易成本基于漂移后weight和目标weight的差
        turnover = np.sum(np.abs(action - drifted_weights))
        cost = turnover * self.transaction_cost
        
        # portfolio return
        portfolio_return = np.dot(drifted_weights, log_ret_with_cash) - cost
        
        # 更新portfolio value和weights
        self.portfolio_value *= np.exp(portfolio_return)
        self.weights = action.copy()
        
        # reward
        reward = portfolio_return
        
        # 移到下一天
        self.t += 1
        done = self.t >= self.T
        
        if done:
            obs = np.zeros(self.n_features + self.n_actions, dtype=np.float32)
        else:
            obs = self._get_obs()
            
        return obs, reward, done, False, {}
    
    def render(self):
        print(f"t={self.t}, portfolio_value={self.portfolio_value:.4f}, weights={self.weights}")