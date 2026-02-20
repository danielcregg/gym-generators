# Gym Generators

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?style=flat-square&logo=openai&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-FF6F00?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

A custom [Gymnasium](https://gymnasium.farama.org/) environment for training reinforcement learning agents to solve the **Combined Economic and Emission Dispatch (CEED)** problem. The agent learns to minimize fuel cost, emissions, and constraint violations across a 10-generator, 24-hour dispatch scheduling problem.

## Overview

This project implements a multi-objective CEED problem as a Gymnasium environment. A reinforcement learning agent controls 9 generators (with 1 slack generator) over a 24-hour period, optimizing power output to meet hourly demand while minimizing operating costs, emissions, and transmission losses. The environment uses linear scalarization to balance competing objectives.

## Features

- **Gymnasium Compatible** -- Fully compatible with Gymnasium's modern API (`step` returns 5-tuple, `reset` accepts `seed`)
- **10-Generator System** -- Realistic generator characteristics including min/max power, ramp-up/down rates, and cost coefficients
- **24-Hour Scheduling** -- Hourly power demand profiles for day-ahead dispatch planning
- **Multi-Objective Optimization** -- Balances fuel cost, emissions, and constraint penalties via weighted scalarization
- **Rich Observation Space** -- Vector observations with normalized hour, generator index, power levels, demand, and historical cost/emissions
- **Dense Rewards** -- Intermediate per-generator cost/emission feedback, plus full-hour cooperative rewards
- **Transmission Loss Modeling** -- B-coefficient loss matrix for realistic power flow calculations
- **Constrained Action Space** -- Enforces generator ramp-rate limits and capacity constraints
- **PPO Training Script** -- Ready-to-use training script with Stable Baselines3
- **Comprehensive Tests** -- Full pytest test suite with 30+ test cases

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

Install directly from GitHub:

```bash
pip install -e "git+https://github.com/danielcregg/gym-generators#egg=gym_generators"
```

Or clone and install locally:

```bash
git clone https://github.com/danielcregg/gym-generators.git
cd gym-generators
pip install -e .
```

Install all dependencies (including training and testing):

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import gymnasium as gym
import gym_generators

env = gym.make("Generators-v0")
obs, info = env.reset(seed=42)

terminated = False
truncated = False
total_reward = 0.0

while not (terminated or truncated):
    action = env.action_space.sample()  # Replace with trained agent
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Episode complete. Total reward: {total_reward:.2f}")
env.close()
```

## Training with PPO

Train a PPO agent using Stable Baselines3:

```bash
# Install training dependencies
pip install stable-baselines3 matplotlib

# Train the agent (default: 50k timesteps)
python train_ppo.py --timesteps 100000

# The trained model and training curves are saved to ./trained_models/
```

The training script:
- Creates a PPO agent with tuned hyperparameters
- Logs episode rewards during training
- Saves the trained model to disk
- Generates training curve plots
- Evaluates the trained model and prints results

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_generators_env.py::TestFullEpisode -v
```

## Environment Details

| Property | Value |
|---|---|
| Observation Space | `Box(-1, 1, shape=(15,), dtype=float32)` |
| Action Space | `Discrete(101)` |
| Generators | 10 (1 slack + 9 agent-controlled) |
| Hours | 24 |
| Steps per Episode | 216 (9 generators x 24 hours) |
| Reward | Negative weighted cost + emissions + penalties |

### Observation Vector (15 dimensions)

| Index | Description |
|---|---|
| 0 | Normalized current hour (0 to 1) |
| 1 | Normalized current generator index (0 to 1) |
| 2-11 | Normalized power levels of all 10 generators |
| 12 | Normalized current demand |
| 13 | Normalized previous hour cost |
| 14 | Normalized previous hour emissions |

### Reward Structure

The reward is the negative weighted sum of three objectives:
- **Fuel Cost** (weight 0.225): Valve-point cost function with sinusoidal ripple
- **Emissions** (weight 0.275): Quadratic + exponential emission function
- **Penalty** (weight 0.5): Large penalty for constraint violations (capacity, ramp rates)

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Gymnasium | Reinforcement learning environment framework |
| Stable Baselines3 | PPO training algorithm |
| NumPy | Numerical computation |
| Pandas | Generator data and demand management |
| pytest | Testing framework |
| Matplotlib | Training curve visualization |

## Bug Fixes from Original

This version fixes several critical bugs from the original implementation:

1. **Unreachable termination**: The episode-end check `n==11 and m==M` was unreachable because `n` was reset to 2 before the check
2. **Indexing inconsistency**: Three different formulas were used for `states_array` indexing; standardized to `(n-1)*M + (m-1)`
3. **B-matrix math bug**: `P_n * B[n][j] * P_n` was used instead of `P_n * B[n][j] * P_j` in the slack generator quadratic
4. **Shared mutable state**: `states_array` and `p_n_m_df` were class variables mutated per-instance
5. **Missing observation_space**: Environment lacked `observation_space`, breaking `gym.make()`
6. **Reward zeroing**: `self.reward = 0` erased computed rewards for 8 of 9 steps per hour

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
