"""Interactive Brokers integration via ib_insync.

Requires TWS/IB Gateway running and reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


class IBKRBroker:
    def __init__(self, cfg: IBKRConfig = IBKRConfig()):
        self.cfg = cfg
        self.ib = None
        try:
            from ib_insync import IB

            self.ib = IB()
        except Exception:
            self.ib = None

    def is_ready(self) -> bool:
        return self.ib is not None

    def connect(self) -> bool:
        if not self.ib:
            return False
        self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
        return self.ib.isConnected()

    def place_market_order(self, symbol: str, qty: float, side: str) -> Dict[str, Any]:
        if not self.ib or not self.ib.isConnected():
            return {"status": "not_connected"}
        from ib_insync import Stock, MarketOrder

        contract = Stock(symbol, "SMART", "USD")
        order = MarketOrder(side.upper(), qty)
        trade = self.ib.placeOrder(contract, order)
        return {"status": "submitted", "order_id": trade.order.orderId}
