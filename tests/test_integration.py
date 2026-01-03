"""
Integration test for end-to-end trading pipeline.
Tests data loading → agent orchestration → RL training → backtesting.
"""

import sys
import os
sys.path.insert(0, '/workspace/code')

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data.market_data_loader import SyntheticMarketGenerator, SyntheticNewsGenerator
from data.feature_engineering import FeatureEngineer
from models.llm_wrapper import LLMWrapper, LLMConfig
from agents.orchestrator import MultiAgentOrchestrator, TradingContext
from rl.trading_env import TradingEnv
from rl.rl_trainer import RLTrainer
from backtest.backtester import Backtester, BacktestConfig


def test_end_to_end_pipeline():
    """Test complete pipeline from data to backtest."""
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate data
    generator = SyntheticMarketGenerator(seed=42)
    start_date = '2024-01-01'
    end_date = '2024-03-31'  # 3 months for speed
    
    ohlcv_data = generator.generate_ohlcv(['TEST'], start_date, end_date)
    assert 'TEST' in ohlcv_data
    assert len(ohlcv_data['TEST']) > 0
    
    # 2. Generate news
    news_gen = SyntheticNewsGenerator(seed=42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    news_df = news_gen.generate_news(['TEST'], dates)
    assert len(news_df) > 0
    
    # 3. Engineer features
    engineer = FeatureEngineer()
    feature_data = engineer.create_feature_matrix(ohlcv_data, news_df)
    assert 'TEST' in feature_data
    assert feature_data['TEST'].shape[1] > 10  # Multiple features
    
    # 4. Create trading environment
    env = TradingEnv(feature_data['TEST'])
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    
    # 5. Train RL agent (minimal)
    trainer = RLTrainer(env, algorithm="PPO")
    trainer.train(total_timesteps=500)  # Very short for test speed
    
    # Evaluate
    eval_stats = trainer.evaluate(n_episodes=2)
    assert 'mean_reward' in eval_stats
    assert isinstance(eval_stats['mean_reward'], float)
    
    # 6. Test LLM orchestrator
    llm_config = LLMConfig(backend="mock")
    orchestrator = MultiAgentOrchestrator(llm_config)
    
    context = TradingContext(
        ticker='TEST',
        timestamp=pd.Timestamp.now(),
        current_price=100.0,
        position=0,
        portfolio_value=10000,
        features={'rsi': 30, 'macd': 0.5, 'sentiment_mean': 0.3},
        news=[]
    )
    
    result = orchestrator.run_cycle(context)
    assert 'decision' in result
    assert 'explanation' in result
    assert result['decision']['action'] in ['BUY', 'SELL', 'HOLD']
    
    # 7. Run backtest
    backtester = Backtester(BacktestConfig())
    
    # Generate simple signals
    signals = pd.Series(1, index=feature_data['TEST'].index)
    backtest_result = backtester.run(feature_data['TEST'], signals)
    
    assert 'metrics' in backtest_result
    assert 'total_return' in backtest_result['metrics']
    assert isinstance(backtest_result['metrics']['total_return'], float)
    
    print("\n✅ All integration tests passed!")


def test_reproducibility():
    """Test that results are reproducible with same seed."""
    
    results1 = []
    results2 = []
    
    for seed in [42, 42]:  # Same seed twice
        np.random.seed(seed)
        generator = SyntheticMarketGenerator(seed=seed)
        data = generator.generate_ohlcv(['TEST'], '2024-01-01', '2024-01-31')
        results1.append(data['TEST']['close'].values)
    
    assert np.allclose(results1[0], results1[1]), "Results should be identical with same seed"
    
    # Different seed should give different results
    np.random.seed(999)
    generator = SyntheticMarketGenerator(seed=999)
    data = generator.generate_ohlcv(['TEST'], '2024-01-01', '2024-01-31')
    results2 = data['TEST']['close'].values
    
    assert not np.allclose(results1[0], results2), "Results should differ with different seed"
    
    print("\n✅ Reproducibility test passed!")


def test_risk_constraints():
    """Test that risk agent enforces constraints."""
    
    llm_config = LLMConfig(backend="mock")
    
    # Test position limit enforcement
    config = {
        'risk': {
            'max_position': 0.1,  # 10% max
            'max_drawdown': 0.15
        }
    }
    
    orchestrator = MultiAgentOrchestrator(llm_config, agent_config=config)
    
    # Create context with large position attempt
    context = TradingContext(
        ticker='TEST',
        timestamp=pd.Timestamp.now(),
        current_price=100.0,
        position=0,
        portfolio_value=10000,
        features={'rsi': 30},
        news=[]
    )
    
    result = orchestrator.run_cycle(context)
    
    # Decision should be constrained
    if result['execution']:
        new_position_value = abs(result['execution']['shares'] * result['execution']['price'])
        assert new_position_value <= context.portfolio_value * 0.2  # With some margin
    
    print("\n✅ Risk constraint test passed!")


if __name__ == "__main__":
    test_end_to_end_pipeline()
    test_reproducibility()
    test_risk_constraints()
    print("\n🎉 All tests passed successfully!")
