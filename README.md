# LLM-Powered Multi-Agent Frameworks for Algorithmic Trading

## 📊 Quick Pilot Results

**Experiment Configuration:**

- **Tickers**: AAPL, MSFT (2 assets)
- **Period**: 2024-01-01 to 2024-12-31 (1 year)
- **Data**: Synthetic (GBM with realistic statistical properties)
- **RL Training**: 5,000 timesteps per agent (PPO algorithm)
- **Runtime**: 127.5 seconds (single CPU)

### Performance Metrics (Hybrid LLM+RL Agent)

| Ticker   | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
| -------- | ------------ | ------------ | ------------ | -------- | ------ |
| **AAPL** | -4.04%       | -1.80        | -4.66%       | 42.41%   | 316    |
| **MSFT** | -4.19%       | -2.13        | -5.34%       | 37.34%   | 316    |

**Note**: Negative returns are expected in this quick pilot due to:

1. **Limited training** (5k timesteps vs. 100k+ for publication results)
2. **Synthetic data** (no real market alpha signals)
3. **Mock LLM backend** (deterministic responses, not adaptive)
4. **No hyperparameter tuning**

**For publication-quality results**, run the full-scale experiment (see below).

---

## 📁 Repository Structure

```
llm-trading-research/
├── code/                          # Source code (10,000+ lines)
│   ├── agents/                   # Multi-agent orchestration
│   │   └── orchestrator.py       # Agent communication & decision flow
│   ├── data/                     # Data infrastructure
│   │   ├── market_data_loader.py # OHLCV, news, macro data
│   │   └── feature_engineering.py # Technical indicators & features
│   ├── models/                   # LLM integration
│   │   └── llm_wrapper.py        # Unified LLM API (OpenAI/Anthropic/Local)
│   ├── rl/                       # Reinforcement learning
│   │   ├── trading_env.py        # Gymnasium trading environment
│   │   └── rl_trainer.py         # PPO/DQN training with SB3
│   ├── backtest/                 # Backtesting engine
│   │   └── backtester.py         # Vectorized backtest + statistics
│   ├── explainability/           # Transparency & interpretability
│   ├── utils/                    # Utilities & visualization
│   └── run_experiment.py         # Main experiment runner
├── data/                          # Data storage
│   ├── raw/                      # Raw market data
│   ├── processed/                # Engineered features
│   └── synthetic/                # Generated datasets
├── figures/                       # Publication figures (6 total)
│   ├── architecture_diagram.png  # System architecture
│   ├── sequence_diagram.png      # Agent interaction flow
│   ├── equity_curves.png         # Portfolio performance
│   ├── drawdown_analysis.png     # Risk analysis
│   ├── performance_comparison.png # Strategy comparison
│   └── explainability_example.png # Decision explanation
├── results/                       # Experimental results
│   ├── metrics/                  # Performance metrics (JSON/CSV)
│   ├── logs/                     # Execution logs
│   └── checkpoints/              # Trained model weights
├── tests/                         # Unit & integration tests
├── docker/                        # Docker configuration
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies (pinned)
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build container
docker-compose build

# Run quick pilot (2 minutes)
docker-compose run llm-trading python code/run_experiment.py

# Run full experiment (4-8 hours, requires GPU)
docker-compose run llm-trading python code/run_experiment.py --config configs/full_experiment.yaml

# View results
docker-compose run llm-trading python -m http.server 8000 --directory results/
# Open browser: http://localhost:8000
```

### Option 2: Local Python Environment

```bash
# Create environment
conda create -n llm-trading python=3.10
conda activate llm-trading

# Install dependencies
pip install -r requirements.txt

# Run experiment
cd code
python run_experiment.py

# Results will be in ../results/ and ../figures/
```

---

## 🔧 Configuration

### Quick Pilot (Default)

```yaml
seed: 42
data:
  source: synthetic # Fast, no API keys needed
  tickers: [AAPL, MSFT]
  start_date: "2024-01-01"
  end_date: "2024-12-31"
llm:
  backend: mock # Deterministic, no API costs
rl:
  timesteps: 5000 # ~2 minutes training
```

### Full Experiment (Publication Quality)

```yaml
seed: 42
data:
  source: yahoo # Real market data
  tickers: [AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM] # 8 assets
  start_date: "2020-01-01"
  end_date: "2024-12-31"
llm:
  backend: openai # GPT-4 for reasoning
  model_name: gpt-4-turbo
rl:
  timesteps: 100000 # ~4-8 hours with GPU
```

**API Keys Required for Full Experiment:**

```bash
export OPENAI_API_KEY="sk-..."          # For GPT-4 reasoning
export FRED_API_KEY="..."               # For macro data (free)
export WANDB_API_KEY="..."              # For experiment tracking (optional)
```

---

## 📈 Experiment Outputs

### 1. Performance Metrics (`results/metrics/`)

- `experiment_results.json`: Complete results with all metrics
- `strategy_comparison.csv`: Side-by-side strategy comparison
- Statistical test results (bootstrap CI, p-values)

### 2. Visualizations (`figures/`)

All figures are **generated from actual experiment outputs**, not mock-ups:

1. **Architecture Diagram**: System components and data flow
2. **Sequence Diagram**: Agent interaction timeline
3. **Equity Curves**: Portfolio value over time vs. buy-and-hold
4. **Performance Comparison**: Bar charts of returns/Sharpe/drawdown
5. **Drawdown Analysis**: Maximum drawdown visualization
6. **Explainability Example**: Sample decision explanation

### 3. Model Checkpoints (`results/checkpoints/`)

- Trained RL policies (PPO models)
- Agent conversation logs
- TensorBoard training curves

---

## 🔬 Experimental Design

### Baselines Implemented

1. **Buy-and-Hold**: Passive benchmark
2. **RL-Only**: Pure reinforcement learning (no LLM)
3. **LLM-Only**: Pure LLM reasoning (no RL)
4. **Hybrid (Proposed)**: LLM reasoning + RL execution

### Ablations Tested

- **Agent Ablations**: Removing individual agents (Analyst, Risk, etc.)
- **Feature Ablations**: Technical-only, Sentiment-only, Macro-only
- **LLM Ablations**: Different model sizes (GPT-3.5 vs GPT-4)

### Statistical Validation

- **Time-series CV**: Walk-forward validation (70/30 train/test split)
- **Bootstrap CI**: 1,000 samples, 95% confidence intervals
- **Paired tests**: Sharpe ratio difference significance

---

## 🧪 Testing

### Run Tests

```bash
# Unit tests
pytest tests/ -v

# Integration test (end-to-end pipeline)
pytest tests/test_integration.py -v

# Coverage report
pytest tests/ --cov=code --cov-report=html
```

### Test Coverage

- Data loading & feature engineering: ✅
- Agent orchestration & messaging: ✅
- RL environment step function: ✅
- Backtesting engine: ✅
- Statistical tests: ✅

---

## 📊 Data Sources & Licensing

### Real Data Sources (Full Experiment)

| Source               | Data Type        | License                 | API Key Required |
| -------------------- | ---------------- | ----------------------- | ---------------- |
| Yahoo Finance        | OHLCV            | Free for non-commercial | No               |
| FRED (St. Louis Fed) | Macro indicators | Public domain           | Yes (free)       |
| SEC EDGAR            | Company filings  | Public domain           | No               |

### Synthetic Data (Quick Pilot)

When real data is unavailable, the system generates **statistically realistic** synthetic data:

- **Price generation**: Geometric Brownian Motion with jumps
- **Correlation structure**: Multivariate normal with configurable correlation (ρ=0.5)
- **News generation**: Template-based with sentiment labels
- **Validation**: Matches real market summary statistics (see `data/README.md`)

**All data generation is deterministic** (seed=42) for reproducibility.

---

## 🔄 Reproducibility

### Exact Environment

```bash
# Option 1: Docker (guaranteed reproducibility)
docker-compose build  # Uses Dockerfile with pinned versions

# Option 2: Conda export
conda env export > environment.yml

# Option 3: pip freeze
pip freeze > requirements_frozen.txt
```

### Seeds & Determinism

All randomness is controlled:

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
env.reset(seed=SEED)
```

### Compute Requirements

| Experiment      | CPU     | GPU      | RAM  | Time   |
| --------------- | ------- | -------- | ---- | ------ |
| Quick Pilot     | 4 cores | Optional | 8GB  | 2 min  |
| Full (No GPU)   | 8 cores | No       | 16GB | 12 hrs |
| Full (With GPU) | 4 cores | RTX 3090 | 16GB | 4 hrs  |

**Estimated Cost (AWS)**:

- Quick Pilot: $0.01 (CPU spot instance)
- Full Experiment: $2-5 (GPU spot instance)

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size in config
rl:
  batch_size: 32  # Default is 64
```

**2. API Rate Limits (OpenAI)**

```bash
# Use local model or increase retry delays
llm:
  backend: local
  model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**3. Slow Training**

```bash
# Use smaller dataset or reduce timesteps
rl:
  timesteps: 10000  # Instead of 100000
```

---
