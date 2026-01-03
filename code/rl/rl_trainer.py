"""
RL agent trainer using Stable-Baselines3.
Integrates with LLM agents for hybrid decision-making.
"""

import os
import logging
from typing import Dict, Optional, Any
import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """Custom callback for logging trading metrics."""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    if self.verbose > 0:
                        logger.info(f"Episode reward: {info['episode']['r']:.2f}, length: {info['episode']['l']}")
        
        return True


class RLTrainer:
    """Trainer for RL trading agents."""
    
    def __init__(
        self,
        env: gym.Env,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        hyperparams: Optional[Dict] = None,
        save_dir: str = "results/checkpoints"
    ):
        """
        Initialize RL trainer.
        
        Args:
            env: Trading environment
            algorithm: RL algorithm (PPO, DQN, A2C)
            policy: Policy network type
            hyperparams: Algorithm hyperparameters
            save_dir: Directory to save checkpoints
        """
        self.env = env
        self.algorithm_name = algorithm
        self.policy = policy
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Default hyperparameters
        default_params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "verbose": 1,
            "tensorboard_log": f"{save_dir}/tensorboard"
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        self.hyperparams = default_params
        
        # Initialize algorithm
        self.model = self._create_model()
        
        logger.info(f"Initialized {algorithm} trainer with policy {policy}")
    
    def _create_model(self):
        """Create RL model."""
        if self.algorithm_name == "PPO":
            return PPO(
                self.policy,
                self.env,
                **self.hyperparams
            )
        elif self.algorithm_name == "DQN":
            return DQN(
                self.policy,
                self.env,
                **self.hyperparams
            )
        elif self.algorithm_name == "A2C":
            return A2C(
                self.policy,
                self.env,
                **self.hyperparams
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def train(
        self,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        callback: Optional[BaseCallback] = None
    ) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            callback: Custom callback
        
        Returns:
            Training statistics
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Create callbacks
        callbacks = []
        
        if callback is None:
            callback = TradingCallback(log_dir=self.save_dir)
        callbacks.append(callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=f"{self.save_dir}/best_model",
            log_path=f"{self.save_dir}/eval",
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        callbacks.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        self.model.save(f"{self.save_dir}/final_model")
        logger.info(f"Training complete. Model saved to {self.save_dir}")
        
        return {
            "total_timesteps": total_timesteps,
            "save_dir": self.save_dir
        }
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        }
    
    def load(self, path: str):
        """Load trained model."""
        if self.algorithm_name == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm_name == "DQN":
            self.model = DQN.load(path, env=self.env)
        elif self.algorithm_name == "A2C":
            self.model = A2C.load(path, env=self.env)
        
        logger.info(f"Loaded model from {path}")


class HybridAgent:
    """
    Hybrid agent combining LLM reasoning with RL policy.
    LLM provides high-level strategy, RL optimizes execution.
    """
    
    def __init__(
        self,
        rl_model,
        llm_orchestrator,
        combination_mode: str = "weighted"  # "weighted", "rl_only", "llm_only"
    ):
        """
        Initialize hybrid agent.
        
        Args:
            rl_model: Trained RL model
            llm_orchestrator: LLM multi-agent orchestrator
            combination_mode: How to combine LLM and RL outputs
        """
        self.rl_model = rl_model
        self.llm_orchestrator = llm_orchestrator
        self.combination_mode = combination_mode
        self.alpha = 0.5  # Weight for LLM signal in weighted mode
    
    def predict(self, observation, context) -> int:
        """
        Make trading decision using hybrid approach.
        
        Args:
            observation: Environment observation for RL
            context: Trading context for LLM
        
        Returns:
            Action (discrete)
        """
        if self.combination_mode == "rl_only":
            action, _ = self.rl_model.predict(observation, deterministic=True)
            return int(action)
        
        elif self.combination_mode == "llm_only":
            result = self.llm_orchestrator.run_cycle(context)
            decision = result['decision']
            
            # Map LLM decision to discrete action
            if decision['action'] == 'BUY':
                return 2
            elif decision['action'] == 'SELL':
                return 0
            else:
                return 1
        
        else:  # weighted combination
            # Get RL action
            rl_action, _ = self.rl_model.predict(observation, deterministic=True)
            rl_action = int(rl_action)
            
            # Get LLM action
            result = self.llm_orchestrator.run_cycle(context)
            decision = result['decision']
            
            if decision['action'] == 'BUY':
                llm_action = 2
            elif decision['action'] == 'SELL':
                llm_action = 0
            else:
                llm_action = 1
            
            # Weighted combination (simple voting)
            if rl_action == llm_action:
                return rl_action
            else:
                # Break tie using alpha weight
                import random
                return llm_action if random.random() < self.alpha else rl_action


if __name__ == "__main__":
    from rl.trading_env import TradingEnv
    from data.market_data_loader import SyntheticMarketGenerator
    from data.feature_engineering import FeatureEngineer
    from datetime import datetime, timedelta
    
    # Generate test data
    generator = SyntheticMarketGenerator(seed=42)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    ohlcv_data = generator.generate_ohlcv(['TEST'], start_date, end_date)
    engineer = FeatureEngineer()
    feature_data = engineer.create_feature_matrix(ohlcv_data)
    
    # Create environment
    env = TradingEnv(feature_data['TEST'])
    
    # Train agent
    trainer = RLTrainer(env, algorithm="PPO")
    stats = trainer.train(total_timesteps=10000)
    
    print(f"\nTraining stats: {stats}")
    
    # Evaluate
    eval_stats = trainer.evaluate(n_episodes=5)
    print(f"\nEvaluation stats:")
    for key, value in eval_stats.items():
        print(f"  {key}: {value:.2f}")
