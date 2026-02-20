# Gym Generators

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenAI Gym](https://img.shields.io/badge/OpenAI_Gym-0081A5?style=flat-square&logo=openai&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

> **Note:** This repository is a fork of [apoddar573/Tic-Tac-Toe-Gym_Environment](https://github.com/apoddar573/Tic-Tac-Toe-Gym_Environment), adapted for power generator optimization.

A custom OpenAI Gym environment for training reinforcement learning agents to efficiently manage and schedule power generators. The agent learns to minimize fuel cost, emissions, and constraint violations across a 10-generator, 24-hour dispatch problem.

## Overview

This project implements a multi-objective combined economic and emission dispatch (CEED) problem as an OpenAI Gym environment. A reinforcement learning agent controls 9 generators (with 1 slack generator) over a 24-hour period, optimizing power output to meet hourly demand while minimizing operating costs, emissions, and transmission losses. The environment uses linear scalarization to balance competing objectives.

## Features

- **Custom Gym Environment** -- Fully compatible with OpenAI Gym's standard interface (`step`, `reset`, `render`)
- **10-Generator System** -- Realistic generator characteristics including min/max power, ramp-up/down rates, and cost coefficients
- **24-Hour Scheduling** -- Hourly power demand profiles for day-ahead dispatch planning
- **Multi-Objective Optimization** -- Balances fuel cost, emissions, and constraint penalties via weighted scalarization
- **Transmission Loss Modeling** -- B-coefficient loss matrix for realistic power flow calculations
- **Constrained Action Space** -- Enforces generator ramp-rate limits and capacity constraints
- **Slack Generator** -- Automatic calculation of slack bus power to meet demand-supply balance
- **Jupyter Notebook Demo** -- Includes a test notebook for quick experimentation

## Prerequisites

- Python 3.6 or higher
- pip package manager
- Jupyter Notebook (optional, for running the demo)

## Getting Started

### Installation

Install directly from GitHub:

```bash
pip install -e git+https://github.com/danielcregg/gym-generators#egg=gym-generators
```

Or clone and install locally:

```bash
git clone https://github.com/danielcregg/gym-generators.git
cd gym-generators
pip install -e .
```

### Usage

```python
import gym
import gym_generators

env = gym.make("Generators-v0")
state = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Replace with trained agent's action
    state, reward, done, info = env.step(action)

print("Episode complete.")
```

The environment registers as `Generators-v0` and supports the standard Gym interface. Each action is a discrete value from 0 to 100 representing a percentage of the generator's power range.

## Tech Stack

| Technology | Purpose                                |
|------------|----------------------------------------|
| Python     | Core programming language              |
| OpenAI Gym | Reinforcement learning framework       |
| NumPy      | Numerical computation                  |
| Pandas     | Generator data and demand management   |
| Jupyter    | Interactive experimentation            |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
