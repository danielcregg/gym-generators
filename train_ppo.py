#!/usr/bin/env python3
"""
PPO Training Script for the Generators-v0 CEED Environment.

This script trains a Proximal Policy Optimization (PPO) agent from Stable Baselines3
on the Combined Economic and Emission Dispatch environment. It saves the trained model
and produces training curves.

Usage:
    python train_ppo.py [--timesteps N] [--eval-episodes N]

Requirements:
    pip install gymnasium stable-baselines3 numpy pandas matplotlib
    pip install -e .   (to install the gym_generators package)
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Ensure the environment is registered
import gym_generators  # noqa: F401


class RewardLoggerCallback(BaseCallback):
    """Custom callback to log episode rewards during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        # Accumulate reward
        rewards = self.locals.get("rewards", [0])
        dones = self.locals.get("dones", [False])

        for reward, done in zip(rewards, dones):
            self._current_reward += reward
            self._current_length += 1
            if done:
                self.episode_rewards.append(self._current_reward)
                self.episode_lengths.append(self._current_length)
                self._current_reward = 0.0
                self._current_length = 0
        return True


def train(total_timesteps=50000, eval_episodes=5, save_dir="./trained_models"):
    """Train a PPO agent on the Generators-v0 environment.

    Args:
        total_timesteps: Total number of environment steps for training.
        eval_episodes: Number of episodes for final evaluation.
        save_dir: Directory to save the trained model and plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("PPO Training for Generators-v0 (CEED)")
    print("=" * 60)

    # Create vectorized environment
    env = make_vec_env("Generators-v0", n_envs=1)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=42,
    )

    # Train with reward logging
    callback = RewardLoggerCallback()
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model_path = os.path.join(save_dir, "ppo_generators")
    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")

    # Plot training curve
    if callback.episode_rewards:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        episodes = range(1, len(callback.episode_rewards) + 1)
        ax1.plot(episodes, callback.episode_rewards, alpha=0.3, color="blue", label="Raw")

        # Smoothed rewards (rolling average)
        window = min(10, len(callback.episode_rewards))
        if window > 1:
            smoothed = np.convolve(
                callback.episode_rewards, np.ones(window) / window, mode="valid"
            )
            ax1.plot(
                range(window, len(callback.episode_rewards) + 1),
                smoothed,
                color="red",
                linewidth=2,
                label=f"Smoothed (window={window})",
            )
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Training Reward Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(episodes, callback.episode_lengths, alpha=0.5, color="green")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Length")
        ax2.set_title("Episode Length Over Training")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved to {plot_path}")
        plt.close()

    # Evaluate
    print(f"\nEvaluating trained model over {eval_episodes} episodes...")
    eval_env = gym.make("Generators-v0")
    total_rewards = []
    total_costs = []
    total_emissions = []

    for ep in range(eval_episodes):
        obs, info = eval_env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        ep_emissions = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if "hourly_cost" in info:
                ep_cost += info["hourly_cost"]
            if "hourly_emissions" in info:
                ep_emissions += info["hourly_emissions"]

        total_rewards.append(ep_reward)
        total_costs.append(ep_cost)
        total_emissions.append(ep_emissions)
        print(f"  Episode {ep + 1}: reward={ep_reward:.2f}, cost={ep_cost:.2f}, emissions={ep_emissions:.2f}")

    eval_env.close()
    env.close()

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Mean reward:    {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Mean cost:      {np.mean(total_costs):.2f} +/- {np.std(total_costs):.2f}")
    print(f"  Mean emissions: {np.mean(total_emissions):.2f} +/- {np.std(total_emissions):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Generators-v0")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--save-dir", type=str, default="./trained_models", help="Save directory")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
    )
