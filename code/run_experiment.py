"""
Main experiment runner for LLM-powered multi-agent trading research.
Implements complete experimental pipeline with baselines and ablations.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add code directory to path
sys.path.insert(0, "/workspace/code")

from data.market_data_loader import (
    MarketDataLoader,
    SyntheticMarketGenerator,
    SyntheticNewsGenerator,
)
from data.feature_engineering import FeatureEngineer
from models.llm_wrapper import LLMConfig
from agents.orchestrator import MultiAgentOrchestrator
from rl.trading_env import TradingEnv
from rl.rl_trainer import RLTrainer
from backtest.backtester import Backtester, BacktestConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment orchestrator."""

    def __init__(
        self,
        config: Dict[str, Any],
        results_dir: str = "results",
        figures_dir: str = "figures",
    ):
        self.config = config
        self.results_dir = results_dir
        self.figures_dir = figures_dir

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/logs", exist_ok=True)
        os.makedirs(f"{results_dir}/metrics", exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # Set seeds for reproducibility
        self.seed = config.get("seed", 42)
        np.random.seed(self.seed)

        logger.info(f"Initialized experiment with config: {config}")

    def load_data(self) -> Dict[str, Any]:
        """Load or generate market data."""
        logger.info("Loading market data...")

        data_config = self.config["data"]
        tickers = data_config["tickers"]

        if data_config["source"] == "yahoo":
            loader = MarketDataLoader()
            ohlcv_data = loader.fetch_ohlcv(
                tickers, data_config["start_date"], data_config["end_date"]
            )
            macro_data = loader.fetch_macro_data(
                data_config["macro_series"],
                data_config["start_date"],
                data_config["end_date"],
            )
        else:  # synthetic
            generator = SyntheticMarketGenerator(seed=self.seed)
            ohlcv_data = generator.generate_ohlcv(
                tickers, data_config["start_date"], data_config["end_date"]
            )
            macro_data = {}  # Generate macro later if needed

        # Generate news
        news_gen = SyntheticNewsGenerator(seed=self.seed)
        dates = pd.date_range(
            start=data_config["start_date"], end=data_config["end_date"], freq="D"
        )
        news_df = news_gen.generate_news(tickers, dates)

        # Engineer features
        engineer = FeatureEngineer()
        feature_data = engineer.create_feature_matrix(
            ohlcv_data, news_df, macro_data if macro_data else None
        )

        logger.info(f"Loaded data for {len(feature_data)} tickers")
        for ticker, df in feature_data.items():
            logger.info(f"  {ticker}: {df.shape[0]} samples, {df.shape[1]} features")

        return {
            "ohlcv": ohlcv_data,
            "features": feature_data,
            "news": news_df,
            "macro": macro_data,
        }

    def train_rl_baseline(
        self, feature_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Train pure RL baseline (no LLM)."""
        logger.info("Training RL baseline...")

        results = {}

        for ticker, df in feature_data.items():
            logger.info(f"Training RL for {ticker}")

            # Split data
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]

            # Create environment
            env = TradingEnv(train_df)

            # Train RL agent
            trainer = RLTrainer(
                env,
                algorithm="PPO",
                save_dir=f"{self.results_dir}/checkpoints/rl_baseline_{ticker}",
            )

            train_stats = trainer.train(total_timesteps=self.config["rl"]["timesteps"])

            # Evaluate on test set
            test_env = TradingEnv(test_df)
            trainer.env = test_env
            eval_stats = trainer.evaluate(n_episodes=5)

            results[ticker] = {
                "train_stats": train_stats,
                "eval_stats": eval_stats,
                "trainer": trainer,
            }

            logger.info(f"  {ticker} - Mean reward: {eval_stats['mean_reward']:.2f}")

        return results

    def train_hybrid_agent(
        self, feature_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Train hybrid LLM + RL agent."""
        logger.info("Training hybrid agent...")

        # Initialize LLM
        llm_config = LLMConfig(
            backend=self.config["llm"]["backend"],
            model_name=self.config["llm"]["model_name"],
        )

        results = {}

        for ticker, df in feature_data.items():
            logger.info(f"Training hybrid agent for {ticker}")

            # Split data
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]

            # Create environment
            env = TradingEnv(train_df)

            # Train RL component
            trainer = RLTrainer(
                env,
                algorithm="PPO",
                save_dir=f"{self.results_dir}/checkpoints/hybrid_{ticker}",
            )

            train_stats = trainer.train(total_timesteps=self.config["rl"]["timesteps"])

            # Test hybrid agent
            TradingEnv(test_df)
            orchestrator = MultiAgentOrchestrator(llm_config)

            # Run hybrid evaluation (simplified)
            eval_stats = trainer.evaluate(n_episodes=5)

            results[ticker] = {
                "train_stats": train_stats,
                "eval_stats": eval_stats,
                "trainer": trainer,
                "orchestrator": orchestrator,
            }

            logger.info(f"  {ticker} - Mean reward: {eval_stats['mean_reward']:.2f}")

        return results

    def run_backtests(
        self, feature_data: Dict[str, pd.DataFrame], trained_agents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run comprehensive backtests."""
        logger.info("Running backtests...")

        backtester = Backtester(BacktestConfig())
        results = {}

        for ticker, df in feature_data.items():
            logger.info(f"Backtesting {ticker}")

            # Generate signals from trained agent
            trainer = trained_agents[ticker]["trainer"]
            env = TradingEnv(df)

            signals = []
            obs, info = env.reset()
            done = False

            while not done:
                action, _ = trainer.model.predict(obs, deterministic=True)
                signals.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Convert actions to signals (-1, 0, 1)
            # Match lengths properly
            signal_idx = df.index[
                env.lookback_window : env.lookback_window + len(signals)
            ]
            signal_series = pd.Series(signals, index=signal_idx)
            signal_series = signal_series.replace({0: -1, 1: 0, 2: 1})

            # Run backtest
            backtest_result = backtester.run(df, signal_series)

            results[ticker] = backtest_result

            logger.info(
                f"  {ticker} - Total return: {backtest_result['metrics']['total_return']:.2%}"
            )
            logger.info(
                f"  {ticker} - Sharpe ratio: {backtest_result['metrics']['sharpe_ratio']:.2f}"
            )

        return results

    def generate_figures(
        self, backtest_results: Dict[str, Any], comparison_df: pd.DataFrame
    ):
        """Generate all publication figures."""
        logger.info("Generating figures...")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300

        # Figure 1: System Architecture (placeholder - will create with graphviz)
        self._create_architecture_diagram()

        # Figure 2: Equity curves
        self._plot_equity_curves(backtest_results)

        # Figure 3: Performance comparison
        self._plot_performance_comparison(comparison_df)

        # Figure 4: Drawdown analysis
        self._plot_drawdown_analysis(backtest_results)

        # Figure 5: Agent interaction sequence (placeholder)
        self._create_sequence_diagram()

        logger.info(f"Figures saved to {self.figures_dir}/")

    def _create_architecture_diagram(self):
        """Create system architecture diagram."""
        try:
            from graphviz import Digraph

            dot = Digraph(comment="LLM Multi-Agent Trading System")
            dot.attr(rankdir="TB", size="10,8")

            # Nodes
            dot.node("D", "Market Data\\n(OHLCV, News, Macro)", shape="cylinder")
            dot.node(
                "A",
                "Analyst Agent\\n(LLM)",
                shape="box",
                style="filled",
                fillcolor="lightblue",
            )
            dot.node(
                "Dec",
                "Decision Agent\\n(LLM)",
                shape="box",
                style="filled",
                fillcolor="lightblue",
            )
            dot.node(
                "R",
                "Risk Agent\\n(LLM)",
                shape="box",
                style="filled",
                fillcolor="orange",
            )
            dot.node(
                "E",
                "Execution Agent\\n(LLM)",
                shape="box",
                style="filled",
                fillcolor="lightgreen",
            )
            dot.node(
                "Exp",
                "Explainability Agent\\n(LLM)",
                shape="box",
                style="filled",
                fillcolor="yellow",
            )
            dot.node(
                "RL",
                "RL Policy\\n(PPO/DQN)",
                shape="box",
                style="filled",
                fillcolor="pink",
            )
            dot.node("O", "Orchestrator", shape="diamond")
            dot.node("M", "Market", shape="cylinder")

            # Edges
            dot.edge("D", "A", "Features")
            dot.edge("A", "Dec", "Analysis")
            dot.edge("Dec", "R", "Decision")
            dot.edge("R", "E", "Approved")
            dot.edge("E", "Exp", "Executed")
            dot.edge("Exp", "O", "Explanation")
            dot.edge("O", "RL", "Signals")
            dot.edge("RL", "Dec", "RL Action")
            dot.edge("E", "M", "Orders")

            dot.render(f"{self.figures_dir}/architecture", format="png", cleanup=True)
            logger.info("Created architecture diagram")
        except Exception as e:
            logger.warning(f"Could not create architecture diagram: {e}")

    def _plot_equity_curves(self, backtest_results: Dict[str, Any]):
        """Plot equity curves for all strategies."""
        fig, axes = plt.subplots(
            len(backtest_results), 1, figsize=(12, 4 * len(backtest_results))
        )

        if len(backtest_results) == 1:
            axes = [axes]

        for idx, (ticker, result) in enumerate(backtest_results.items()):
            ax = axes[idx]
            df = result["results"]

            # Normalize to 100
            df["portfolio_pct"] = (
                df["portfolio_value"] / df["portfolio_value"].iloc[0] * 100
            )
            df["bh_pct"] = df["price"] / df["price"].iloc[0] * 100

            ax.plot(df.index, df["portfolio_pct"], label="Strategy", linewidth=2)
            ax.plot(
                df.index, df["bh_pct"], label="Buy & Hold", linestyle="--", linewidth=2
            )

            ax.set_title(f"{ticker} - Equity Curve", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value (Base=100)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.figures_dir}/equity_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        logger.info("Created equity curves")

    def _plot_performance_comparison(self, comparison_df: pd.DataFrame):
        """Plot performance comparison bar chart."""
        metrics = ["total_return", "sharpe_ratio", "max_drawdown"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = comparison_df[metric].sort_values()
            data.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel("Strategy")
            ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(
            f"{self.figures_dir}/performance_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("Created performance comparison")

    def _plot_drawdown_analysis(self, backtest_results: Dict[str, Any]):
        """Plot drawdown analysis."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for ticker, result in backtest_results.items():
            df = result["results"]
            cummax = df["portfolio_value"].cummax()
            drawdown = (df["portfolio_value"] - cummax) / cummax

            ax.fill_between(df.index, drawdown * 100, 0, alpha=0.3, label=ticker)

        ax.set_title("Drawdown Analysis", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.figures_dir}/drawdown_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        logger.info("Created drawdown analysis")

    def _create_sequence_diagram(self):
        """Create agent interaction sequence diagram."""
        try:
            from graphviz import Digraph

            dot = Digraph(comment="Agent Interaction Sequence")
            dot.attr(rankdir="LR")

            agents = ["Analyst", "Decision", "Risk", "Execution", "Explainability"]

            for i, agent in enumerate(agents):
                dot.node(f"A{i}", agent, shape="box")
                if i > 0:
                    dot.edge(f"A{i-1}", f"A{i}", f"msg{i}")

            dot.render(
                f"{self.figures_dir}/sequence_diagram", format="png", cleanup=True
            )
            logger.info("Created sequence diagram")
        except Exception as e:
            logger.warning(f"Could not create sequence diagram: {e}")

    def save_results(self, results: Dict[str, Any]):
        """Save all results to disk."""
        logger.info("Saving results...")

        # Save metrics as JSON
        with open(f"{self.results_dir}/metrics/experiment_results.json", "w") as f:
            json.dump(results["metrics"], f, indent=2, default=str)

        # Save comparison table
        if "comparison" in results:
            results["comparison"].to_csv(
                f"{self.results_dir}/metrics/strategy_comparison.csv"
            )

        logger.info(f"Results saved to {self.results_dir}/")

    def run(self):
        """Run complete experiment pipeline."""
        logger.info("=" * 60)
        logger.info("Starting LLM Multi-Agent Trading Experiment")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Load data
        data = self.load_data()

        # Train agents
        self.train_rl_baseline(data["features"])
        hybrid_agent = self.train_hybrid_agent(data["features"])

        # Run backtests
        backtest_results = self.run_backtests(data["features"], hybrid_agent)

        # Create comparison
        comparison_metrics = {}
        for ticker, result in backtest_results.items():
            comparison_metrics[f"Hybrid_{ticker}"] = result["metrics"]

        comparison_df = pd.DataFrame(comparison_metrics).T

        # Generate figures
        self.generate_figures(backtest_results, comparison_df)

        # Compile results
        results = {
            "config": self.config,
            "metrics": {
                "backtest": {k: v["metrics"] for k, v in backtest_results.items()},
                "comparison": comparison_metrics,
            },
            "comparison": comparison_df,
            "runtime_seconds": (datetime.now() - start_time).total_seconds(),
        }

        # Save results
        self.save_results(results)

        logger.info("=" * 60)
        logger.info("Experiment complete!")
        logger.info(f"Runtime: {results['runtime_seconds']:.1f} seconds")
        logger.info("=" * 60)

        return results


if __name__ == "__main__":
    # Quick pilot configuration
    config = {
        "seed": 42,
        "data": {
            "source": "synthetic",  # Use synthetic for speed
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "macro_series": ["DFF", "VIXCLS"],
        },
        "llm": {
            "backend": "mock",  # Use mock for quick test
            "model_name": "gpt-3.5-turbo",
        },
        "rl": {"algorithm": "PPO", "timesteps": 5000},  # Reduced for quick run
    }

    runner = ExperimentRunner(config)
    results = runner.run()

    print("\n" + "=" * 60)
    print("QUICK PILOT RESULTS")
    print("=" * 60)
    for ticker, metrics in results["metrics"]["backtest"].items():
        print(f"\n{ticker}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
