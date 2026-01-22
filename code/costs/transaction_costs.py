"""Transaction cost model.

Includes:
- Commission (per share)
- Bid-ask spread (bps)
- Slippage (bps)
- Market impact (square-root model)

Outputs per-trade total costs and net execution price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class CostModelConfig:
    commission_per_share: float = 0.005  # $/share
    spread_bps: float = 1.0  # bps of mid
    slippage_bps: float = 2.0
    impact_coef: float = 0.10  # scale factor for sqrt impact
    adv_shares: float = 5_000_000  # average daily volume shares (fallback)


class TransactionCostModel:
    def __init__(self, config: CostModelConfig):
        self.config = config

    def estimate(
        self,
        side: str,
        mid_price: float,
        shares: float,
        volatility_daily: float = 0.02,
        adv_shares: float | None = None,
    ) -> Dict[str, Any]:
        side = side.upper()
        adv = float(adv_shares if adv_shares is not None else self.config.adv_shares)
        shares_abs = float(abs(shares))

        # Spread: half-spread paid
        half_spread = (self.config.spread_bps / 1e4) * mid_price / 2
        # Slippage: linear
        slip = (self.config.slippage_bps / 1e4) * mid_price
        # Impact: sqrt(order/ADV) * vol * price
        participation = 0.0 if adv <= 0 else shares_abs / adv
        impact = (
            self.config.impact_coef
            * np.sqrt(max(participation, 0.0))
            * volatility_daily
            * mid_price
        )

        # Directional: buys pay up, sells receive less
        directional = half_spread + slip + impact
        if side == "BUY":
            exec_price = mid_price + directional
        elif side == "SELL":
            exec_price = mid_price - directional
        else:
            exec_price = mid_price

        commission = self.config.commission_per_share * shares_abs
        cost_dollars = commission + shares_abs * (exec_price - mid_price) * (
            1 if side == "BUY" else -1
        )

        return {
            "mid_price": mid_price,
            "exec_price": float(exec_price),
            "shares": float(shares),
            "commission": float(commission),
            "half_spread_$": float(half_spread),
            "slippage_$": float(slip),
            "impact_$": float(impact),
            "total_cost_$": float(cost_dollars),
        }
