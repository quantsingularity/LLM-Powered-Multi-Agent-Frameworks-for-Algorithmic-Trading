"""
Simplified integration test focusing on core functionality.
"""

import sys
sys.path.insert(0, '/home/user/llm-trading-research/code')

import numpy as np
import pandas as pd

from data.market_data_loader import SyntheticMarketGenerator
from data.feature_engineering import FeatureEngineer
from rl.trading_env import TradingEnv
from rl.rl_trainer import RLTrainer
from backtest.backtester import Backtester, BacktestConfig

print("="*60)
print("INTEGRATION TEST: End-to-End Trading Pipeline")
print("="*60)

# Set seed
np.random.seed(42)

# 1. Data Generation
print("\n[1/5] Generating synthetic market data...")
generator = SyntheticMarketGenerator(seed=42)
ohlcv_data = generator.generate_ohlcv(['TEST'], '2024-01-01', '2024-03-31')
assert 'TEST' in ohlcv_data
assert len(ohlcv_data['TEST']) > 50
print(f"✓ Generated {len(ohlcv_data['TEST'])} days of OHLCV data")

# 2. Feature Engineering
print("\n[2/5] Engineering features...")
engineer = FeatureEngineer()
feature_data = engineer.create_feature_matrix(ohlcv_data)
assert 'TEST' in feature_data
n_features = feature_data['TEST'].shape[1]
print(f"✓ Created {n_features} features")

# 3. RL Environment
print("\n[3/5] Setting up RL environment...")
env = TradingEnv(feature_data['TEST'])
obs, info = env.reset()
assert obs.shape == env.observation_space.shape
print(f"✓ Environment initialized (obs_dim={obs.shape[0]})")

# Test random episode
total_reward = 0
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print(f"✓ Random episode completed (total_reward={total_reward:.2f})")

# 4. RL Training
print("\n[4/5] Training RL agent...")
trainer = RLTrainer(env, algorithm="PPO")
trainer.train(total_timesteps=500, eval_freq=10000)  # Minimal training
print("✓ Training completed")

# 5. Backtesting
print("\n[5/5] Running backtest...")
backtester = Backtester(BacktestConfig())

# Generate signals from trained agent
test_env = TradingEnv(feature_data['TEST'])
obs, info = test_env.reset()
signals = []

done = False
while not done:
    action, _ = trainer.model.predict(obs, deterministic=True)
    signals.append(int(action))
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

# Run backtest
signal_series = pd.Series(signals, index=feature_data['TEST'].index[env.lookback_window:env.lookback_window+len(signals)])
signal_series = signal_series.replace({0: -1, 1: 0, 2: 1})

result = backtester.run(feature_data['TEST'], signal_series)
metrics = result['metrics']

print(f"✓ Backtest completed:")
print(f"  - Total Return: {metrics['total_return']:.2%}")
print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  - Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"  - Number of Trades: {int(metrics['num_trades'])}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
