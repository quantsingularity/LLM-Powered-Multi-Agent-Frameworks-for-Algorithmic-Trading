"""Backtesting engine with realistic market simulation.

Enhancements:
- Uses TransactionCostModel for bid-ask, slippage, commissions, impact
- Reports gross and net returns
- Supports continuous target weights
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

from costs.transaction_costs import TransactionCostModel, CostModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    position_limit: float = 0.2
    risk_free_rate: float = 0.02
    cost_model: CostModelConfig = CostModelConfig()


class Backtester:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.costs = TransactionCostModel(self.config.cost_model)

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        price_col: str = "close",
        adv_col: Optional[str] = None,
        vol_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Running backtest on {len(data)} periods")

        prices = data[price_col].values
        signals = signals.reindex(data.index, fill_value=0.0).values

        n = len(prices)
        cash = np.zeros(n)
        position = np.zeros(n)
        pv_gross = np.zeros(n)
        pv_net = np.zeros(n)
        trades = []

        cash[0] = self.config.initial_capital
        pv_gross[0] = cash[0]
        pv_net[0] = cash[0]

        for t in range(1, n):
            cash[t] = cash[t - 1]
            position[t] = position[t - 1]

            signal = float(signals[t - 1])
            mid = float(prices[t])

            # gross PV at mid
            pv_gross[t] = cash[t] + position[t] * mid
            pv_net[t] = pv_gross[t]

            target_value = signal * self.config.position_limit * pv_net[t]
            target_shares = target_value / mid
            shares_to_trade = target_shares - position[t]

            if abs(shares_to_trade) > 1e-6:
                side = "BUY" if shares_to_trade > 0 else "SELL"

                adv = (
                    float(data.iloc[t][adv_col])
                    if adv_col and adv_col in data.columns
                    else None
                )
                vol = (
                    float(data.iloc[t][vol_col])
                    if vol_col and vol_col in data.columns
                    else 0.02
                )

                cost_info = self.costs.estimate(
                    side, mid, shares_to_trade, volatility_daily=vol, adv_shares=adv
                )
                exec_price = cost_info["exec_price"]
                commission = cost_info["commission"]

                abs(shares_to_trade) * exec_price

                if side == "BUY":
                    required_cash = shares_to_trade * exec_price + commission
                    if required_cash <= cash[t] + 1e-9:
                        cash[t] -= required_cash
                        position[t] += shares_to_trade
                    else:
                        # can't afford; skip
                        continue
                else:
                    cash[t] -= (
                        shares_to_trade * exec_price - commission
                    )  # shares_to_trade negative
                    position[t] += shares_to_trade

                pv_net[t] = cash[t] + position[t] * mid

                trades.append(
                    {
                        "time": data.index[t],
                        "side": side,
                        "shares": float(abs(shares_to_trade)),
                        "mid": mid,
                        "exec_price": float(exec_price),
                        **cost_info,
                        "cash": float(cash[t]),
                        "position": float(position[t]),
                        "pv_net": float(pv_net[t]),
                    }
                )

        metrics = {
            "gross": self._calculate_metrics(pv_gross, data.index),
            "net": self._calculate_metrics(pv_net, data.index),
        }

        results_df = pd.DataFrame(
            {
                "cash": cash,
                "position": position,
                "pv_gross": pv_gross,
                "pv_net": pv_net,
                "price": prices,
            },
            index=data.index,
        )

        return {
            "results": results_df,
            "metrics": metrics,
            "trades": trades,
            "config": self.config,
        }

    def _calculate_metrics(
        self, pv: np.ndarray, idx: pd.DatetimeIndex
    ) -> Dict[str, float]:
        r = np.diff(pv) / pv[:-1]
        r = r[np.isfinite(r)]
        total_return = (pv[-1] / pv[0]) - 1
        n_years = (idx[-1] - idx[0]).days / 365.25
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = np.std(r) * np.sqrt(252) if len(r) > 1 else 0
        sharpe = (
            (ann_return - self.config.risk_free_rate) / ann_vol if ann_vol > 0 else 0
        )
        peak = np.maximum.accumulate(pv)
        dd = (pv / peak) - 1
        mdd = float(dd.min())
        return {
            "total_return": float(total_return),
            "annualized_return": float(ann_return),
            "volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(mdd),
            "final_value": float(pv[-1]),
        }
