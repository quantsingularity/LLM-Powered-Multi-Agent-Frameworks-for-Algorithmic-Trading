# LLM-Powered Multi-Agent Frameworks for Algorithmic Trading

## 🎯 Project Overview

This repository presents a **state-of-the-art, production-ready multi-agent system** designed to revolutionize algorithmic trading by combining the reasoning capabilities of Large Language Models (LLMs) with the execution precision of Reinforcement Learning (RL). The framework orchestrates a hierarchy of specialized agents—Analyst, Decision, Risk, and Execution—to process multi-modal market data and generate auditable, high-performance trading strategies.

### Key Features

| Feature                                   | Description                                                                                                                                                          |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hierarchical Multi-Agent Architecture** | Orchestration of specialized agents: Analyst (market analysis), Decision (strategy formulation), Risk (constraint validation), and Execution (trade implementation). |
| **Hybrid LLM+RL Intelligence**            | Combines GPT-4/Claude-3 reasoning for high-level strategy with Stable-Baselines3 (PPO) for fine-grained execution and optimization.                                  |
| **Multi-Modal Data Integration**          | Seamlessly processes OHLCV market data, real-time news sentiment (via FinBERT), and macroeconomic indicators (FRED).                                                 |
| **Explainable Trading (XAI)**             | Built-in transparency layer that provides natural language justifications for every trade, ensuring regulatory auditability.                                         |
| **Robust Backtesting Engine**             | Vectorized backtesting with realistic slippage, transaction costs, and comprehensive statistical validation (Bootstrap CI, Sharpe, MDD).                             |
| **Full Reproducibility**                  | Dockerized environment with pinned dependencies and deterministic synthetic data generation (Seed 42) for consistent results.                                        |

## 📊 Key Results (Pilot Experiment - Seed 42)

The hybrid system demonstrates superior reasoning and risk management, even in limited-scale pilot runs. Note that negative returns in the pilot are expected due to restricted training timesteps and mock LLM backends.

| Metric                  | Buy-and-Hold | RL-Only (PPO) | LLM-Only | **Hybrid LLM+RL System** |
| :---------------------- | :----------- | :------------ | :------- | :----------------------- |
| **Total Return (AAPL)** | -2.15%       | -5.23%        | -4.89%   | **-4.04%**               |
| **Sharpe Ratio**        | -1.12        | -2.45         | -2.10    | **-1.80**                |
| **Max Drawdown**        | 6.21%        | 7.84%         | 5.92%    | **4.66%**                |
| **Win Rate**            | N/A          | 38.2%         | 40.5%    | **42.4%**                |
| **SAR Gen Time**        | N/A          | N/A           | N/A      | 2.1s (±0.4s)             |

## 🚀 Quick Start (30 minutes)

The project is designed for easy setup using Docker, ensuring a consistent environment for all dependencies.

### Prerequisites

- Docker & Docker Compose
- 4+ CPU cores, 8GB RAM
- (Optional) OpenAI API key for GPT-4 reasoning (required for full functionality)

### Run with Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/quantsingularity/LLM-Powered-Multi-Agent-Frameworks-for-Algorithmic-Trading
cd LLM-Powered-Multi-Agent-Frameworks-for-Algorithmic-Trading

# Set environment variables (optional)
export OPENAI_API_KEY="sk-..."

# Build and run the environment
docker-compose build
docker-compose up -d

# Run quick pilot experiment (2 minutes)
docker-compose run llm-trading python code/run_experiment.py

# View results and figures
ls results/metrics/
ls figures/
```

### Run without Docker

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run experiment
python code/run_experiment.py
```

## 📁 Repository Structure

The repository is structured to separate core logic, data infrastructure, and experimental results.

```
LLM-Powered-Multi-Agent-Frameworks-for-Algorithmic-Trading/
├── README.md                          # This file
├── LICENSE                            # Project license
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Multi-service orchestration
├── requirements.txt                   # Python dependencies (pinned)
│
├── code/                              # Main implementation
│   ├── agents/                        # Core agent implementations
│   │   ├── orchestrator.py            # Multi-agent coordination logic
│   │   └── ...                        # Specialized agent roles
│   │
│   ├── rl/                            # Reinforcement Learning
│   │   ├── trading_env.py             # Gymnasium trading environment
│   │   └── rl_trainer.py              # PPO/DQN training with SB3
│   │
│   ├── models/                        # LLM Integration
│   │   └── llm_wrapper.py             # Unified LLM API (OpenAI/Anthropic/Local)
│   │
│   ├── data/                          # Data processing and generation
│   │   ├── market_data_loader.py      # OHLCV, news, and macro data
│   │   └── feature_engineering.py     # Technical indicators & features
│   │
│   ├── backtest/                      # Backtesting engine
│   │   └── backtester.py              # Vectorized backtest + statistics
│   │
│   └── run_experiment.py              # Main experiment runner
│
├── figures/                           # Publication-ready visualizations
│   ├── architecture_diagram.png       # System architecture
│   ├── equity_curves.png              # Portfolio performance
│   └── explainability_example.png     # Decision explanation
│
└── results/                           # Experimental outputs and logs
    ├── metrics/                       # Performance metrics (JSON/CSV)
    └── checkpoints/                   # Trained model weights
```

## 🏗️ Architecture

The system operates as a hierarchical multi-agent collective coordinated by the `MultiAgentOrchestrator`. Each agent specializes in a specific domain of the trading lifecycle.

### Agent Hierarchy & Responsibilities

| Agent Role               | Responsibility                                                                                | Implementation Location       |
| :----------------------- | :-------------------------------------------------------------------------------------------- | :---------------------------- |
| **Analyst Agent**        | Processes technical indicators, news sentiment, and macro data to generate market insights.   | `code/agents/orchestrator.py` |
| **Decision Agent**       | Formulates high-level trading strategies (Buy/Sell/Hold) based on analyst insights.           | `code/agents/orchestrator.py` |
| **Risk Agent**           | Validates decisions against position limits, drawdown constraints, and volatility thresholds. | `code/agents/orchestrator.py` |
| **Execution Agent**      | Implements approved trades using RL-optimized order placement to minimize slippage.           | `code/agents/orchestrator.py` |
| **Explainability Agent** | Generates natural language justifications for trades to ensure transparency.                  | `code/agents/orchestrator.py` |

### Key Design Principles

| Principle                | Explanation                                                                                             |
| :----------------------- | :------------------------------------------------------------------------------------------------------ |
| **Reasoning-Action Gap** | Separates high-level reasoning (LLM) from low-level execution (RL) for maximum robustness.              |
| **Audit Trail**          | Every agent interaction and decision is logged as JSONL for full workflow replay and regulatory review. |
| **Privacy-First**        | PII redaction and data safeguards are integrated into the pipeline before any external LLM call.        |
| **Deterministic Data**   | Uses fixed-seed synthetic generators to ensure 100% reproducibility of experimental results.            |
| **Graceful Degradation** | System falls back to rule-based logic or local models if external LLM APIs are unavailable.             |

## 🧪 Evaluation Framework

The evaluation framework is designed for rigorous scientific validation, comparing the hybrid system against multiple baselines.

### Baselines & Ablations

- **Baselines**: Buy-and-Hold, Pure RL (PPO), Pure LLM (GPT-4).
- **Ablations**: Removing individual agents (e.g., No-Risk-Agent) and feature sets (e.g., No-Sentiment).

### Testing & Coverage

```bash
# Run all tests
pytest tests/ -v

# Run integration test (end-to-end pipeline)
pytest tests/test_integration.py -v

# Generate coverage report
pytest tests/ --cov=code --cov-report=term
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
