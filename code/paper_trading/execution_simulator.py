"""Paper trading order execution simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd

from costs.transaction_costs import TransactionCostModel, CostModelConfig


@dataclass
class PaperConfig:
    initial_cash: float = 100000.0
    cost_model: CostModelConfig = CostModelConfig()


class PaperBroker:
    def __init__(self, cfg: PaperConfig):
        self.cfg = cfg
        self.costs = TransactionCostModel(cfg.cost_model)
        self.cash = cfg.initial_cash
        self.positions: Dict[str, float] = {}
        self.trades: list[dict] = []

    def place_market_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        mid_price: float,
        timestamp: Optional[pd.Timestamp] = None,
        volatility_daily: float = 0.02,
        adv_shares: Optional[float] = None,
    ) -> Dict[str, Any]:
        side = side.upper()
        ts = timestamp or pd.Timestamp.utcnow()
        sign = 1 if side == "BUY" else -1

        cost = self.costs.estimate(
            side,
            mid_price,
            qty,
            volatility_daily=volatility_daily,
            adv_shares=adv_shares,
        )
        exec_price = cost["exec_price"]
        commission = cost["commission"]

        notional = qty * exec_price
        cash_delta = -(sign * notional) - commission

        if side == "BUY" and self.cash + cash_delta < -1e-9:
            return {"status": "rejected", "reason": "insufficient_cash"}

        self.cash += cash_delta
        self.positions[ticker] = self.positions.get(ticker, 0.0) + sign * qty

        trade = {
            "timestamp": ts,
            "ticker": ticker,
            "side": side,
            "qty": float(qty),
            "mid": float(mid_price),
            "exec_price": float(exec_price),
            "commission": float(commission),
            "total_cost_$": float(cost["total_cost_$"]),
            "cash": float(self.cash),
            "position": float(self.positions[ticker]),
        }
        self.trades.append(trade)
        return {"status": "filled", "trade": trade}

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        pv = self.cash
        for t, q in self.positions.items():
            pv += q * float(prices.get(t, 0.0))
        return float(pv)
