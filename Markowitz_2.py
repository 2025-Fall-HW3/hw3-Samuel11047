"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation
"""

class MyPortfolio:
    """
    Strategy: Global Minimum Variance (GMV)
    Rationale: Mathematically minimizes portfolio volatility using Quadratic Programming (Gurobi).
               Lower volatility (denominator) directly boosts the Sharpe Ratio.
    """

    def __init__(self, price, exclude, lookback=252, top_n=None, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # 1. 取得資產列表 (排除 SPY)
        assets = self.price.columns[self.price.columns != self.exclude]
        
        # 2. 定義資產數量 (關鍵修正: 確保變數存在)
        n_assets = len(assets)

        # 初始化權重 DataFrame
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 滾動計算
        for i in range(self.lookback + 1, len(self.price)):
            current_date = self.price.index[i]

            # 取得回溯期內的回報數據
            R_n = self.returns[assets].iloc[i - self.lookback : i]
            
            # 計算協方差矩陣 (Covariance Matrix)
            Sigma = R_n.cov().values
            
            # 使用 Gurobi 求解最小變異數組合
            # 目標: Minimize (w' * Sigma * w) -> 最小化風險
            try:
                with gp.Env(empty=True) as env:
                    env.setParam("OutputFlag", 0)  # 關閉輸出
                    env.setParam("DualReductions", 0)
                    env.start()
                    
                    with gp.Model(env=env, name="min_variance") as model:
                        # 定義變數 (權重 w >= 0)
                        w = model.addMVar(n_assets, name="w", lb=0.0)
                        
                        # 設定目標函數: 最小化變異數
                        model.setObjective(w @ Sigma @ w, gp.GRB.MINIMIZE)
                        
                        # 設定約束: 權重總和為 1
                        model.addConstr(w.sum() == 1, name="budget")
                        
                        # 優化
                        model.optimize()
                        
                        if model.status == gp.GRB.OPTIMAL:
                            optimal_weights = w.X
                        else:
                            # 優化失敗時的備案：等權重
                            optimal_weights = np.ones(n_assets) / n_assets

                    # 寫入權重
                    self.portfolio_weights.loc[current_date, assets] = optimal_weights
            
            except Exception:
                # 發生任何錯誤時的備案：等權重
                self.portfolio_weights.loc[current_date, assets] = 1.0 / n_assets

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_