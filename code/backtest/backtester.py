"""
Backtesting engine with realistic market simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    position_limit: float = 0.2  # Max 20% in single position
    max_leverage: float = 1.0  # No leverage
    risk_free_rate: float = 0.02  # 2% annual


class Backtester:
    """Vectorized backtesting engine."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,  # Trading signals (-1, 0, 1) or continuous
        price_col: str = "close",
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with price data
            signals: Series of trading signals (index-aligned with data)
            price_col: Column name for prices

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest on {len(data)} periods")

        # Initialize
        prices = data[price_col].values
        signals = signals.reindex(data.index, fill_value=0).values

        n_periods = len(prices)
        cash = np.zeros(n_periods)
        position = np.zeros(n_periods)
        portfolio_value = np.zeros(n_periods)
        trades = []

        cash[0] = self.config.initial_capital
        position[0] = 0
        portfolio_value[0] = cash[0]

        # Simulate trading
        for t in range(1, n_periods):
            # Previous state
            cash[t] = cash[t - 1]
            position[t] = position[t - 1]

            # Get signal
            signal = signals[t - 1]  # Use previous signal

            # Calculate target position
            current_value = cash[t] + position[t] * prices[t]
            target_position_value = signal * self.config.position_limit * current_value
            target_shares = target_position_value / prices[t]

            # Trade if signal changes significantly
            shares_to_trade = target_shares - position[t]

            if abs(shares_to_trade) > 1e-6:
                # Apply slippage
                if shares_to_trade > 0:
                    exec_price = prices[t] * (1 + self.config.slippage)
                else:
                    exec_price = prices[t] * (1 - self.config.slippage)

                # Calculate cost
                trade_value = abs(shares_to_trade) * exec_price
                cost = trade_value * self.config.transaction_cost

                # Check if we have enough cash
                if shares_to_trade > 0:  # Buy
                    required_cash = shares_to_trade * exec_price + cost
                    if required_cash <= cash[t]:
                        cash[t] -= required_cash
                        position[t] += shares_to_trade

                        trades.append(
                            {
                                "time": data.index[t],
                                "action": "BUY",
                                "shares": shares_to_trade,
                                "price": exec_price,
                                "cost": cost,
                            }
                        )
                else:  # Sell
                    cash[t] -= (
                        shares_to_trade * exec_price - cost
                    )  # Note: shares_to_trade is negative
                    position[t] += shares_to_trade

                    trades.append(
                        {
                            "time": data.index[t],
                            "action": "SELL",
                            "shares": abs(shares_to_trade),
                            "price": exec_price,
                            "cost": cost,
                        }
                    )

            # Update portfolio value
            portfolio_value[t] = cash[t] + position[t] * prices[t]

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_value, data.index)

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "cash": cash,
                "position": position,
                "portfolio_value": portfolio_value,
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
        self, portfolio_values: np.ndarray, timestamps: pd.DatetimeIndex
    ) -> Dict[str, float]:
        """Calculate performance metrics."""

        # Returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[np.isfinite(returns)]

        # Total return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        # Annualized return
        n_years = (timestamps[-1] - timestamps[0]).days / 365.25
        annualized_return = (
            (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        )

        # Volatility
        annual_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # Sharpe ratio
        excess_return = annualized_return - self.config.risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown)

        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        positive_returns = np.sum(returns > 0)
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "num_trades": len(returns),
            "final_value": portfolio_values[-1],
        }

    def compare_strategies(
        self,
        data: pd.DataFrame,
        strategies: Dict[str, pd.Series],
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Compare multiple trading strategies.

        Args:
            data: Price data
            strategies: Dictionary mapping strategy name to signals
            price_col: Price column

        Returns:
            DataFrame with comparison metrics
        """
        results = {}

        for name, signals in strategies.items():
            logger.info(f"Backtesting strategy: {name}")
            backtest_result = self.run(data, signals, price_col)
            results[name] = backtest_result["metrics"]

        # Also add buy-and-hold baseline
        bh_signals = pd.Series(1.0, index=data.index)
        bh_result = self.run(data, bh_signals, price_col)
        results["Buy-and-Hold"] = bh_result["metrics"]

        return pd.DataFrame(results).T


def calculate_bootstrap_confidence_interval(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for Sharpe ratio difference.

    Args:
        strategy_returns: Returns from trading strategy
        baseline_returns: Returns from baseline
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dictionary with CI bounds and p-value
    """
    n_samples = len(strategy_returns)

    # Original Sharpe difference
    sharpe_strategy = (
        np.mean(strategy_returns) / np.std(strategy_returns)
        if np.std(strategy_returns) > 0
        else 0
    )
    sharpe_baseline = (
        np.mean(baseline_returns) / np.std(baseline_returns)
        if np.std(baseline_returns) > 0
        else 0
    )
    sharpe_diff_observed = sharpe_strategy - sharpe_baseline

    # Bootstrap
    sharpe_diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        strategy_sample = strategy_returns[indices]
        baseline_sample = baseline_returns[indices]

        sharpe_s = (
            np.mean(strategy_sample) / np.std(strategy_sample)
            if np.std(strategy_sample) > 0
            else 0
        )
        sharpe_b = (
            np.mean(baseline_sample) / np.std(baseline_sample)
            if np.std(baseline_sample) > 0
            else 0
        )

        sharpe_diffs.append(sharpe_s - sharpe_b)

    sharpe_diffs = np.array(sharpe_diffs)

    # Calculate CI
    alpha = 1 - confidence
    lower = np.percentile(sharpe_diffs, 100 * alpha / 2)
    upper = np.percentile(sharpe_diffs, 100 * (1 - alpha / 2))

    # Calculate p-value (proportion of bootstrap samples where diff <= 0)
    p_value = np.mean(sharpe_diffs <= 0)

    return {
        "sharpe_diff": sharpe_diff_observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "significant": p_value < (1 - confidence),
    }


if __name__ == "__main__":
    from data.market_data_loader import SyntheticMarketGenerator
    from datetime import datetime, timedelta

    # Generate test data
    generator = SyntheticMarketGenerator(seed=42)
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    data = generator.generate_ohlcv(["TEST"], start_date, end_date)["TEST"]

    # Create simple momentum strategy
    data["returns"] = data["close"].pct_change()
    data["momentum"] = data["returns"].rolling(20).mean()
    signals = pd.Series(0, index=data.index)
    signals[data["momentum"] > 0] = 1
    signals[data["momentum"] < 0] = -1

    # Backtest
    backtester = Backtester()
    result = backtester.run(data, signals)

    print("\n=== Backtest Results ===")
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.4f}")

    print(f"\nNumber of trades: {len(result['trades'])}")
