"""Trade explanation report generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd


@dataclass
class TradeReport:
    ticker: str
    start: str
    end: str
    net_metrics: Dict[str, Any]
    gross_metrics: Dict[str, Any]
    trades: List[Dict[str, Any]]
    llm_explanations: List[str]


def to_markdown(r: TradeReport) -> str:
    lines = []
    lines.append(f"# Trade Report: {r.ticker}")
    lines.append(f"Period: {r.start} → {r.end}")
    lines.append("\n## Performance")
    lines.append("\n### Net (after costs)")
    lines.append(pd.DataFrame([r.net_metrics]).to_markdown(index=False))
    lines.append("\n### Gross (before costs)")
    lines.append(pd.DataFrame([r.gross_metrics]).to_markdown(index=False))
    lines.append("\n## Recent Trades")
    if r.trades:
        lines.append(pd.DataFrame(r.trades[-20:]).to_markdown(index=False))
    else:
        lines.append("No trades executed.")
    if r.llm_explanations:
        lines.append("\n## LLM Explanations")
        for i, ex in enumerate(r.llm_explanations[-5:], 1):
            lines.append(f"\n### Explanation {i}\n{ex}")
    return "\n".join(lines)
