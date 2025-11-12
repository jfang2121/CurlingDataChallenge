# CurlingDataChallenge

This repository provides a reinforcement learning environment and training pipeline for self-play curling simulation using Proximal Policy Optimization (PPO).

## Features
- **Custom Curling Environment**: Implements the physics and rules of curling using OpenAI Gym interface, supporting self-play between two teams.
- **PPO Agent**: Actor-Critic neural network and PPO training logic for continuous control.
- **Self-Play Training**: Runs self-play matches between two agents, collects trajectories, and trains the policy.
- **Simulation & Rendering**: Physics simulation for stone movement and collisions, with optional text-based rendering.
- **Results Visualization**: Training script saves reward and score plots for analysis.

## Getting Started
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the PPO agent:
   ```
   python train_ppo_selfplay.py
   ```
   This will run self-play training and save model checkpoints and plots.

## File Overview
- `curling_env.py`: Curling environment for RL, including physics and scoring.
- `ppo_agent.py`: PPO agent and neural network definitions.
- `train_ppo_selfplay.py`: Main training loop for self-play PPO.
- `curlingsim.py`: Physics simulation for curling stones.
- `requirements.txt`: Python dependencies.
- `models/`: Saved model checkpoints.

## Output
- Trained model weights (`ppo_agent_selfplay{n_games}.pth`)
- Reward and score plots (`ppo_rewards_{n_games}.png`, `ppo_scores_{n_games}.png`)

## Usage
You can modify the number of games, hyperparameters, or environment settings in `train_ppo_selfplay.py` to experiment with different training setups.

---
    