"""Quantitative baselines.

Implements signals for:
- Momentum
- Mean-reversion
- Pairs trading (cointegration-based spread z-score)
- Fama-French factor investing (using available factors if provided)

All signal generators return a pd.Series or pd.DataFrame of target weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class MomentumConfig:
    lookback: int = 20


def momentum_signal(
    prices: pd.Series, cfg: MomentumConfig = MomentumConfig()
) -> pd.Series:
    ret = prices.pct_change()
    mom = ret.rolling(cfg.lookback).mean()
    sig = pd.Series(0.0, index=prices.index)
    sig[mom > 0] = 1.0
    sig[mom < 0] = -1.0
    return sig


@dataclass
class MeanReversionConfig:
    lookback: int = 20
    z_entry: float = 1.0


def mean_reversion_signal(
    prices: pd.Series, cfg: MeanReversionConfig = MeanReversionConfig()
) -> pd.Series:
    ma = prices.rolling(cfg.lookback).mean()
    sd = prices.rolling(cfg.lookback).std().replace(0, np.nan)
    z = (prices - ma) / sd
    sig = pd.Series(0.0, index=prices.index)
    sig[z > cfg.z_entry] = -1.0
    sig[z < -cfg.z_entry] = 1.0
    return sig


@dataclass
class PairsConfig:
    lookback: int = 60
    z_entry: float = 1.5


def pairs_trading_signals(
    price_a: pd.Series,
    price_b: pd.Series,
    cfg: PairsConfig = PairsConfig(),
) -> pd.DataFrame:
    df = pd.concat([price_a, price_b], axis=1).dropna()
    df.columns = ["A", "B"]

    # Rolling hedge ratio via OLS
    hedge = pd.Series(index=df.index, dtype=float)
    spread = pd.Series(index=df.index, dtype=float)

    for i in range(cfg.lookback, len(df)):
        window = df.iloc[i - cfg.lookback : i]
        X = sm.add_constant(window["B"])
        y = window["A"]
        res = sm.OLS(y, X).fit()
        beta = float(res.params["B"])
        hedge.iloc[i] = beta
        spread.iloc[i] = df["A"].iloc[i] - beta * df["B"].iloc[i]

    mu = spread.rolling(cfg.lookback).mean()
    sd = spread.rolling(cfg.lookback).std().replace(0, np.nan)
    z = (spread - mu) / sd

    w = pd.DataFrame(0.0, index=df.index, columns=["A", "B"])
    w.loc[z > cfg.z_entry, "A"] = -0.5
    w.loc[z > cfg.z_entry, "B"] = 0.5
    w.loc[z < -cfg.z_entry, "A"] = 0.5
    w.loc[z < -cfg.z_entry, "B"] = -0.5

    return w.reindex(price_a.index).fillna(0.0)


@dataclass
class FamaFrenchConfig:
    # rolling regression window
    lookback: int = 252


def factor_investing_signal(
    returns: pd.Series,
    factors: pd.DataFrame,
    cfg: FamaFrenchConfig = FamaFrenchConfig(),
) -> pd.Series:
    """Estimate rolling alpha vs factors and go long when alpha positive."""
    df = pd.concat([returns, factors], axis=1).dropna()
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    sig = pd.Series(0.0, index=df.index)

    for i in range(cfg.lookback, len(df)):
        y_w = y.iloc[i - cfg.lookback : i]
        X_w = sm.add_constant(X.iloc[i - cfg.lookback : i])
        res = sm.OLS(y_w, X_w).fit()
        alpha = float(res.params.get("const", 0.0))
        sig.iloc[i] = 1.0 if alpha > 0 else -1.0

    return sig.reindex(returns.index).fillna(0.0)
