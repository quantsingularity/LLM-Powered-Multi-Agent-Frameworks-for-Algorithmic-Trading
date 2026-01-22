"""Portfolio-level risk management utilities.

Implements:
- Kelly criterion position sizing (fractional Kelly)
- Max drawdown hard stop
- Sector exposure limits
- Portfolio volatility targeting

These functions are designed to be framework-agnostic and can be used by
LLM agents, RL policies, baselines, and paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    max_drawdown: float = -0.15
    # Fractional Kelly multiplier to reduce estimation error risk
    kelly_fraction: float = 0.5
    # Per-sector max exposure as fraction of portfolio gross exposure
    sector_max_exposure: float = 0.25
    # Target annualized vol for portfolio
    target_vol_annual: float = 0.12
    # Lookback for volatility estimates
    vol_lookback: int = 60
    # Floor/ceiling for position sizing
    max_position_pct: float = 0.20
    min_position_pct: float = 0.0


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

    @staticmethod
    def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
        peak = equity_curve.cummax()
        dd = (equity_curve / peak) - 1.0
        return dd

    def max_drawdown_breached(self, equity_curve: pd.Series) -> bool:
        dd = self.compute_drawdown(equity_curve)
        return float(dd.min()) <= self.config.max_drawdown

    @staticmethod
    def kelly_fraction(mu: float, sigma: float) -> float:
        """Classic Kelly for Gaussian returns: f* = mu / sigma^2.

        mu: expected return per period (same unit as sigma)
        sigma: std dev per period
        """
        if sigma <= 0:
            return 0.0
        return float(mu / (sigma**2))

    def kelly_position_size(
        self,
        expected_return: float,
        volatility: float,
    ) -> float:
        f = self.kelly_fraction(expected_return, volatility)
        f *= self.config.kelly_fraction
        f = float(
            np.clip(f, -self.config.max_position_pct, self.config.max_position_pct)
        )
        if f < self.config.min_position_pct:
            f = 0.0
        return f

    def volatility_target_scaler(
        self,
        portfolio_returns: pd.Series,
    ) -> float:
        """Returns leverage scaler to hit target vol (no leverage >1 enforced here)."""
        r = portfolio_returns.dropna().tail(self.config.vol_lookback)
        if len(r) < 5:
            return 1.0
        realized_vol_daily = float(r.std())
        realized_vol_annual = realized_vol_daily * np.sqrt(252)
        if realized_vol_annual <= 0:
            return 1.0
        scaler = self.config.target_vol_annual / realized_vol_annual
        return float(np.clip(scaler, 0.0, 1.0))

    def enforce_sector_limits(
        self,
        target_weights: Dict[str, float],
        ticker_to_sector: Dict[str, str],
    ) -> Dict[str, float]:
        """Caps total exposure per sector and renormalizes remaining weights."""
        sector_exposure: Dict[str, float] = {}
        for t, w in target_weights.items():
            sec = ticker_to_sector.get(t, "UNKNOWN")
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + abs(w)

        capped = dict(target_weights)
        for sec, exp in sector_exposure.items():
            if exp <= self.config.sector_max_exposure:
                continue
            # Scale down tickers in this sector proportionally
            scale = self.config.sector_max_exposure / exp
            for t, w in list(capped.items()):
                if ticker_to_sector.get(t, "UNKNOWN") == sec:
                    capped[t] = w * scale

        # Renormalize to keep gross exposure <= 1 and within max_position_pct
        gross = sum(abs(w) for w in capped.values())
        if gross > 1e-12 and gross > 1.0:
            capped = {t: w / gross for t, w in capped.items()}

        capped = {
            t: float(
                np.clip(w, -self.config.max_position_pct, self.config.max_position_pct)
            )
            for t, w in capped.items()
        }
        return capped

    def apply_all_controls(
        self,
        desired_weight: float,
        expected_return: float,
        volatility: float,
        equity_curve: Optional[pd.Series] = None,
        portfolio_returns: Optional[pd.Series] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Apply kelly + drawdown + vol targeting to a single-asset weight."""
        reasons: Dict[str, float] = {}

        # Max drawdown hard stop
        if equity_curve is not None and self.max_drawdown_breached(equity_curve):
            reasons["max_drawdown_stop"] = 1.0
            return 0.0, reasons

        # Kelly
        kelly_w = self.kelly_position_size(expected_return, volatility)
        reasons["kelly_weight"] = kelly_w

        w = float(np.clip(desired_weight, -abs(kelly_w), abs(kelly_w)))

        # Vol targeting
        if portfolio_returns is not None:
            scaler = self.volatility_target_scaler(portfolio_returns)
            reasons["vol_target_scaler"] = scaler
            w *= scaler

        w = float(
            np.clip(w, -self.config.max_position_pct, self.config.max_position_pct)
        )
        return w, reasons
