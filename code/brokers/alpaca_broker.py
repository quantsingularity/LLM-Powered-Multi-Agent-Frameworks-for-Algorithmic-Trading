"""Alpaca broker integration.

Uses alpaca-py. Requires:
- ALPACA_API_KEY
- ALPACA_API_SECRET
- ALPACA_PAPER (true/false)

This module is safe to import even if credentials are missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AlpacaConfig:
    paper: bool = True


class AlpacaBroker:
    def __init__(self, cfg: AlpacaConfig = AlpacaConfig()):
        self.cfg = cfg
        self.client = None
        try:
            from alpaca.trading.client import TradingClient

            key = os.getenv("ALPACA_API_KEY")
            secret = os.getenv("ALPACA_API_SECRET")
            if key and secret:
                self.client = TradingClient(key, secret, paper=cfg.paper)
        except Exception:
            self.client = None

    def is_ready(self) -> bool:
        return self.client is not None

    def submit_market_order(self, symbol: str, qty: float, side: str) -> Dict[str, Any]:
        if not self.client:
            return {"status": "not_configured"}
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        o = self.client.submit_order(req)
        return {"status": "submitted", "order_id": getattr(o, "id", None)}
