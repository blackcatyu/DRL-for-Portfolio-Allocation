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
        self.n_features = features.shape[1]
        self.T = len(features)
        
        # action space：3个资产的权重，加上现金，共4个，和为1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # observation space：当天的15个特征 + 当前持仓权重3个
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + self.n_assets,),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        self.t = 1  # 从第1天开始，第0天作为前一天
        self.weights = np.array([1/3, 1/3, 1/3], dtype=np.float32)  # 初始等权重
        self.portfolio_value = 1.0  # 初始净值归一化为1
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        # 当天特征 + 当前持仓权重
        return np.concatenate([self.features[self.t], self.weights]).astype(np.float32)
    
    def step(self, action):
        # 归一化action，确保权重和为1
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # 计算当天的log return
        log_ret = np.log(self.prices[self.t] / self.prices[self.t - 1])
        
        # 计算交易成本：权重变化越大，成本越高
        turnover = np.sum(np.abs(action - self.weights))
        cost = turnover * self.transaction_cost
        
        # 计算portfolio return
        portfolio_return = np.dot(action, log_ret) - cost
        
        # 更新portfolio value和weights
        self.portfolio_value *= np.exp(portfolio_return)
        self.weights = action
        
        # reward先用单步return，后面可以换
        reward = portfolio_return
        
        # 移到下一天
        self.t += 1
        done = self.t >= self.T
        
        if done:
            obs = np.zeros(self.n_features + self.n_assets, dtype=np.float32)
        else:
            obs = self._get_obs()
            
        return obs, reward, done, False, {}
    
    def render(self):
        print(f"t={self.t}, portfolio_value={self.portfolio_value:.4f}")