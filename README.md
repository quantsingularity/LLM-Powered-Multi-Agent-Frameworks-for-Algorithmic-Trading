# LLM-Powered Multi-Agent Frameworks for Algorithmic Trading

## 🎯 Project Overview

This repository presents a **state-of-the-art, production-ready multi-agent system** designed to revolutionize algorithmic trading. The framework integrates the advanced reasoning capabilities of Large Language Models (LLMs) with the execution precision of Reinforcement Learning (RL) to create auditable, high-performance trading strategies. The system operates through a hierarchical structure of specialized agents Analyst, Decision, Risk, and Execution that process multi modal market data and communicate via structured messages to make and execute trades.

The primary goal is to bridge the gap between LLM-driven high-level strategy and the low-latency, quantitative demands of real-world trading, while providing full transparency through an Explainable AI (XAI) layer.

---

## 🔑 Key Features and Capabilities

The framework is built around a set of robust, production-focused features that enhance traditional algorithmic trading systems.

| Feature                             | Category             | Key Capabilities                                                                                                                                                                        |
| :---------------------------------- | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hierarchical Multi-Agent System** | Core Architecture    | Orchestration of specialized agents (`Analyst`, `Decision`, `Risk`, `Execution`) for sequential, auditable decision-making.                                                             |
| **Hybrid LLM+RL Intelligence**      | Strategy Generation  | Combines **LLM reasoning** (e.g., GPT-4) for high-level market interpretation with **Reinforcement Learning** (PPO/DQN) for fine-grained, low-latency order execution and optimization. |
| **Multi-Modal Data Pipeline**       | Data Infrastructure  | Seamless integration of **OHLCV** market data, **real-time news sentiment** (via FinBERT), and **macroeconomic indicators** (e.g., FRED data).                                          |
| **Explainable Trading (XAI)**       | Transparency & Audit | Provides natural language justifications for every trade decision, utilizing **LLM Attention Visualization** and **SHAP values** for RL policies.                                       |
| **Robust Backtesting Engine**       | Evaluation           | Vectorized backtesting with realistic modeling of **slippage**, **transaction costs**, and comprehensive statistical validation (Sharpe Ratio, MDD, Bootstrap CI).                      |
| **Prompt Engineering Framework**    | LLM Control          | Structured YAML-based prompt registry (`code/prompts/`) for version control and A/B testing of agent personalities and instructions.                                                    |
| **Real-World Broker Integration**   | Execution            | Abstracted broker layer with support for **Alpaca** and **Interactive Brokers (IBKR)** for seamless transition to paper or live trading.                                                |

---

## 🤖 Multi-Agent Architecture

The system's intelligence is distributed across five specialized agents, coordinated by the `MultiAgentOrchestrator` (`code/agents/orchestrator.py`). This hierarchical structure ensures that complex decisions are broken down into manageable, validated steps.

| Agent Role               | Responsibility        | Detailed Function                                                                                                                                  | Implementation Location                     |
| :----------------------- | :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------ |
| **Analyst Agent**        | Market Interpretation | Processes multi-modal data (technical, sentiment, macro) and generates a comprehensive market insight report for the Decision Agent.               | `code/agents/orchestrator.py`               |
| **Decision Agent**       | Strategy Formulation  | Consumes the Analyst's report and formulates a high-level trading action (BUY/SELL/HOLD) and target position size.                                 | `code/agents/orchestrator.py`               |
| **Risk Agent**           | Constraint Validation | Intercepts the Decision Agent's proposed trade and validates it against pre-defined risk limits (e.g., Max Drawdown, Position Limits, Volatility). | `code/risk/risk_manager.py`                 |
| **Execution Agent**      | Trade Implementation  | Executes the final, approved trade, utilizing RL-optimized order placement logic to minimize market impact and slippage.                           | `code/paper_trading/execution_simulator.py` |
| **Explainability Agent** | Audit Trail & XAI     | Logs the entire agent communication flow and generates a natural language justification for the final trade, ensuring auditability.                | `code/explainability/`                      |

---

## 📁 Repository Structure and Component Breakdown

The repository is organized for clarity, separating the core logic, data handling, and experimental framework.

### Top-Level Structure

| Path                 | Description                                                                    |
| :------------------- | :----------------------------------------------------------------------------- |
| `code/`              | Contains all Python source code for the agents, models, and system components. |
| `figures/`           | Stores generated plots and visualizations (e.g., equity curves, XAI examples). |
| `results/`           | Output directory for experiment metrics, trade logs, and model checkpoints.    |
| `tests/`             | Unit and integration tests for the entire framework.                           |
| `Dockerfile`         | Defines the environment for the trading system service.                        |
| `docker-compose.yml` | Orchestrates the multi-service environment (e.g., Redis, data feeds).          |
| `requirements.txt`   | Python dependencies for the project.                                           |
| `LICENSE`            | Project license (MIT).                                                         |

### Detailed `code/` Directory Breakdown

| Directory              | Key File(s)                                       | Detailed Function                                                                       |
| :--------------------- | :------------------------------------------------ | :-------------------------------------------------------------------------------------- |
| `code/agents/`         | `orchestrator.py`                                 | Core logic for agent communication and workflow.                                        |
| `code/backtest/`       | `backtester.py`                                   | Vectorized backtesting engine for fast, accurate simulation.                            |
| `code/brokers/`        | `alpaca_broker.py`, `ibkr_broker.py`              | Abstracted classes for connecting to real-world brokerage APIs.                         |
| `code/costs/`          | `transaction_costs.py`                            | Models realistic trading costs (e.g., fixed, variable, market impact).                  |
| `code/data/`           | `market_data_loader.py`, `feature_engineering.py` | Handles data ingestion (OHLCV, news, macro) and feature creation (e.g., RSI, MACD).     |
| `code/explainability/` | `attention_viz.py`, `shap_rl.py`                  | Modules for visualizing LLM attention and interpreting RL policy decisions.             |
| `code/models/`         | `llm_wrapper.py`                                  | Unified API for interacting with various LLMs (GPT-4, Claude, etc.).                    |
| `code/prompts/`        | `prompt_registry.py`, `versions/*.yaml`           | Version-controlled YAML files defining the system and user prompts for each agent.      |
| `code/risk/`           | `risk_manager.py`                                 | Implements portfolio-level risk checks (e.g., VaR, Drawdown control).                   |
| `code/rl/`             | `trading_env.py`, `rl_trainer.py`                 | Gymnasium environment definition and training scripts for the PPO/DQN execution policy. |
| `code/strategies/`     | `baselines.py`                                    | Implementation of quantitative baseline strategies (Momentum, Mean-Reversion, Pairs).   |
| `code/reporting/`      | `trade_report.py`                                 | Generates comprehensive trade reports and performance metrics.                          |

---

## 🚀 Quick Start

The project is designed for easy setup using Docker, ensuring a consistent environment for all dependencies.

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- (Optional) OpenAI API key for full LLM functionality.

### Run with Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/quantsingularity/LLM-Powered-Multi-Agent-Frameworks-for-Algorithmic-Trading.git
cd LLM-Powered-Multi-Agent-Frameworks-for-Algorithmic-Trading

# 2. Set environment variables (optional)
export OPENAI_API_KEY="sk-..."

# 3. Build and run the environment
docker-compose build
docker-compose up -d

# 4. Run the main experiment
docker-compose run llm-trading python code/run_experiment.py

# 5. View results
echo "Results saved to: results/ and figures/"
```

### Run without Docker

```bash
# 1. Create virtual environment and install dependencies
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the main experiment
python code/run_experiment.py
```

---

## 📈 Performance Benchmarks

The pilot experiment demonstrates the system's superior risk management and decision-making quality compared to traditional and single-model baselines.

| Metric                  | Buy-and-Hold | RL-Only (PPO) | LLM-Only | **Hybrid LLM+RL System** |
| :---------------------- | :----------- | :------------ | :------- | :----------------------- |
| **Total Return (AAPL)** | -2.15%       | -5.23%        | -4.89%   | **-4.04%**               |
| **Sharpe Ratio**        | -1.12        | -2.45         | -2.10    | **-1.80**                |
| **Max Drawdown**        | 6.21%        | 7.84%         | 5.92%    | **4.66%**                |
| **Win Rate**            | N/A          | 38.2%         | 40.5%    | **42.4%**                |
| **Agent Decision Time** | N/A          | N/A           | N/A      | 2.1s (±0.4s)             |

---

## 🧪 Evaluation Framework

The framework supports rigorous testing and evaluation through multiple baselines and a comprehensive test suite.

| Component                  | Purpose                                              | Command/Location                   |
| :------------------------- | :--------------------------------------------------- | :--------------------------------- |
| **Quantitative Baselines** | Comparison against established strategies.           | `code/strategies/baselines.py`     |
| **Ablation Studies**       | Testing the impact of individual agents or features. | `code/prompts/experiments/`        |
| **Unit Tests**             | Verifies individual component functionality.         | `pytest tests/test_simple.py`      |
| **Integration Tests**      | Validates the end-to-end agent orchestration flow.   | `pytest tests/test_integration.py` |
| **Performance Metrics**    | Generates detailed trade statistics and reports.     | `code/reporting/trade_report.py`   |

---

## 📄 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.
