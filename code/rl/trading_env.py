"""
Reinforcement Learning environment for LLM-agent trading.
Implements Gymnasium environment with realistic market dynamics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Trading environment compatible with Gymnasium/Stable-Baselines3.
    
    Observation space: Market features + portfolio state
    Action space: Discrete (buy/sell/hold) or Continuous (position size)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,  # 10 bps
        slippage: float = 0.0005,  # 5 bps
        max_position: float = 1.0,  # Max 100% of portfolio in single asset
        lookback_window: int = 20,
        action_type: str = "discrete",  # "discrete" or "continuous"
        reward_scaling: float = 1000.0,
    ):
        """
        Initialize trading environment.
        
        Args:
            data: DataFrame with features and prices (must have 'close' column)
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost rate (% of trade value)
            slippage: Slippage rate (% of price)
            max_position: Maximum position size as fraction of portfolio
            lookback_window: Number of historical steps to include in observation
            action_type: "discrete" (3 actions) or "continuous" (position size)
            reward_scaling: Scale factor for rewards
        """
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.action_type = action_type
        self.reward_scaling = reward_scaling
        
        # Validate data
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Get feature columns (exclude price columns)
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        self.feature_cols = [c for c in data.columns if c not in price_cols]
        self.n_features = len(self.feature_cols)
        
        logger.info(f"Trading environment initialized with {self.n_features} features")
        
        # Normalize features
        self.feature_mean = data[self.feature_cols].mean()
        self.feature_std = data[self.feature_cols].std().replace(0, 1)
        self.data[self.feature_cols] = (data[self.feature_cols] - self.feature_mean) / self.feature_std
        
        # Define action space
        if action_type == "discrete":
            # 0: sell, 1: hold, 2: buy
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous action: target position size [-max_position, max_position]
            self.action_space = spaces.Box(
                low=-max_position,
                high=max_position,
                shape=(1,),
                dtype=np.float32
            )
        
        # Define observation space
        # Features: [market_features (lookback_window x n_features), portfolio_state (3)]
        # Portfolio state: [position, cash_ratio, total_value_ratio]
        obs_dim = lookback_window * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Number of shares held
        self.total_value = initial_balance
        self.portfolio_values = []
        self.trades = []
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_value = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Parse action
        if self.action_type == "discrete":
            target_position = self._discrete_action_to_position(action)
        else:
            target_position = float(action[0])
        
        # Execute trade
        reward = self._execute_trade(target_position, current_price)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = self.total_value <= self.initial_balance * 0.5  # Stop if 50% drawdown
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _discrete_action_to_position(self, action: int) -> float:
        """Convert discrete action to target position."""
        if action == 0:  # Sell
            return -0.5 * self.max_position
        elif action == 1:  # Hold
            return 0.0  # No change
        else:  # Buy
            return 0.5 * self.max_position
    
    def _execute_trade(self, target_position_pct: float, current_price: float) -> float:
        """
        Execute trade and return reward.
        
        Args:
            target_position_pct: Target position as % of portfolio value
            current_price: Current asset price
        
        Returns:
            Reward for this step
        """
        # Calculate target shares
        target_value = target_position_pct * self.total_value
        target_shares = target_value / current_price
        
        # Calculate shares to trade
        shares_to_trade = target_shares - self.position
        
        if abs(shares_to_trade) < 1e-6:
            # No significant trade
            reward = self._calculate_reward(0.0)
            self.portfolio_values.append(self.total_value)
            return reward
        
        # Apply slippage
        if shares_to_trade > 0:
            execution_price = current_price * (1 + self.slippage)
        else:
            execution_price = current_price * (1 - self.slippage)
        
        # Calculate trade cost
        trade_value = abs(shares_to_trade) * execution_price
        cost = trade_value * self.transaction_cost
        
        # Execute trade
        self.balance -= shares_to_trade * execution_price + cost
        self.position += shares_to_trade
        
        # Update total value
        prev_value = self.total_value
        self.total_value = self.balance + self.position * current_price
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'action': 'BUY' if shares_to_trade > 0 else 'SELL',
            'shares': abs(shares_to_trade),
            'price': execution_price,
            'cost': cost,
            'balance': self.balance,
            'position': self.position,
            'total_value': self.total_value
        })
        
        # Calculate reward
        reward = self._calculate_reward(self.total_value - prev_value)
        self.portfolio_values.append(self.total_value)
        
        return reward
    
    def _calculate_reward(self, pnl: float) -> float:
        """
        Calculate reward from P&L.
        
        Reward components:
        1. P&L (scaled)
        2. Risk penalty (volatility)
        3. Transaction cost penalty (implicit in pnl)
        """
        # Base reward: scaled P&L
        reward = (pnl / self.initial_balance) * self.reward_scaling
        
        # Add risk penalty if we have enough history
        if len(self.portfolio_values) > 10:
            returns = np.diff(self.portfolio_values[-10:]) / self.portfolio_values[-10:-1]
            volatility = np.std(returns)
            reward -= volatility * 10  # Penalize high volatility
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Market features (lookback window)
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        feature_data = self.data.iloc[start_idx:end_idx][self.feature_cols].values
        
        # Pad if needed (at start of episode)
        if len(feature_data) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(feature_data), self.n_features))
            feature_data = np.vstack([padding, feature_data])
        
        # Flatten features
        features_flat = feature_data.flatten()
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['close']
        position_value = self.position * current_price
        
        portfolio_state = np.array([
            position_value / self.total_value if self.total_value > 0 else 0,  # Position ratio
            self.balance / self.total_value if self.total_value > 0 else 0,  # Cash ratio
            self.total_value / self.initial_balance  # Total value ratio
        ])
        
        # Combine
        obs = np.concatenate([features_flat, portfolio_state]).astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional info."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'total_value': self.total_value,
            'num_trades': len(self.trades)
        }
    
    def render(self):
        """Render environment state."""
        current_price = self.data.iloc[self.current_step]['close']
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.2f} shares")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Return: {(self.total_value/self.initial_balance - 1)*100:.2f}%")
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Calculate portfolio performance statistics."""
        if len(self.portfolio_values) < 2:
            return {}
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # Total return
        total_return = (values[-1] / values[0]) - 1
        
        # Sharpe ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Win rate
        if len(self.trades) > 0:
            profitable_trades = sum(1 for t in self.trades if t['action'] == 'SELL' and self.position > 0)
            win_rate = profitable_trades / len(self.trades) if len(self.trades) > 0 else 0
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'final_value': values[-1]
        }


class MultiAssetTradingEnv(gym.Env):
    """Trading environment for portfolio of multiple assets."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        **kwargs
    ):
        """
        Initialize multi-asset environment.
        
        Args:
            data_dict: Dictionary mapping ticker to feature DataFrame
            initial_balance: Starting cash
            transaction_cost: Transaction cost rate
        """
        super().__init__()
        
        self.tickers = list(data_dict.keys())
        self.n_assets = len(self.tickers)
        
        # Create individual environments
        self.envs = {
            ticker: TradingEnv(
                data=data,
                initial_balance=initial_balance / self.n_assets,
                transaction_cost=transaction_cost,
                **kwargs
            )
            for ticker, data in data_dict.items()
        }
        
        # Combined action space (action for each asset)
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        
        # Combined observation space
        single_obs_dim = self.envs[self.tickers[0]].observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(single_obs_dim * self.n_assets,),
            dtype=np.float32
        )
        
        logger.info(f"Multi-asset environment initialized with {self.n_assets} assets")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset all environments."""
        observations = []
        infos = {}
        
        for ticker, env in self.envs.items():
            obs, info = env.reset(seed=seed)
            observations.append(obs)
            infos[ticker] = info
        
        return np.concatenate(observations), infos
    
    def step(self, actions: np.ndarray):
        """Step all environments."""
        observations = []
        total_reward = 0.0
        terminated = True
        truncated = False
        infos = {}
        
        for i, ticker in enumerate(self.tickers):
            obs, reward, term, trunc, info = self.envs[ticker].step(actions[i])
            observations.append(obs)
            total_reward += reward
            terminated = terminated and term
            truncated = truncated or trunc
            infos[ticker] = info
        
        return np.concatenate(observations), total_reward, terminated, truncated, infos
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get combined portfolio statistics."""
        stats = {}
        for ticker, env in self.envs.items():
            stats[ticker] = env.get_portfolio_stats()
        
        # Calculate aggregate stats
        total_value = sum(env.total_value for env in self.envs.values())
        total_initial = sum(env.initial_balance for env in self.envs.values())
        
        stats['aggregate'] = {
            'total_return': (total_value / total_initial) - 1,
            'final_value': total_value
        }
        
        return stats


if __name__ == "__main__":
    # Test environment
    from data.market_data_loader import SyntheticMarketGenerator
    from data.feature_engineering import FeatureEngineer
    from datetime import datetime, timedelta
    
    # Generate synthetic data
    generator = SyntheticMarketGenerator(seed=42)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    ohlcv_data = generator.generate_ohlcv(['TEST'], start_date, end_date)
    
    # Engineer features
    engineer = FeatureEngineer()
    feature_data = engineer.create_feature_matrix(ohlcv_data)
    
    # Create environment
    env = TradingEnv(feature_data['TEST'])
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test random episode
    obs, info = env.reset()
    total_reward = 0
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    stats = env.get_portfolio_stats()
    print(f"\nEpisode stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
