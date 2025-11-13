"""
PPO Self-Play Training Script for Curling Environment

This script trains two agents to play curling against each other using
Proximal Policy Optimization (PPO) with self-play.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_dim: int, action_dim: int, alpha, hidden_size: int = 256, 
                 action_bin=(10, 10, 10)):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor head: output logits for each bin per action dimension
        self.actor_logits = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, sum(action_bin))
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Action bounds for the curling environment
        # Speed, angle, spin
        self.action_low = torch.tensor([2.8, 85, -3.0])
        self.action_high = torch.tensor([3.2, 95, 3.0])
        self.n_bins = action_bin  # Number of bins per action dimension
        self.total_bins = sum(self.n_bins)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, obs):
        features = self.shared(obs)
        logits = self.actor_logits(features)
        # Split logits for each action dimension
        split_logits = torch.split(logits, self.n_bins, dim=-1)
        # Create categorical distributions for each action dimension
        dists = [torch.distributions.Categorical(logits=logit) for logit in split_logits]
        value = self.critic(features)
        return dists, value
    
    def get_action(self, dists):
        # Sample discrete bin indices for each action dimension
        bin_indices = [dist.sample() for dist in dists]
        bin_indices_tensor = torch.stack(bin_indices, dim=-1)
        # Map bin indices to physical action values
        action = []
        for i, bins in enumerate(self.n_bins):
            low = self.action_low[i]
            high = self.action_high[i]
            val = low + (high - low) * bin_indices_tensor[..., i].float() / (bins - 1)
            action.append(val)
        action = torch.stack(action, dim=-1)
        return action, bin_indices_tensor

    def get_log_prob(self, dists, bin_indices_tensor):
        # Compute log probability for each action dimension
        log_probs = [dist.log_prob(bin_indices_tensor[..., i]) for i, dist in enumerate(dists)]
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).mean()
        return log_prob, entropy


# ============================================================================
# Trajectory Buffer
# ============================================================================
class TrajectoryBuffer:
    """
    Stores trajectories for one team during a game.
    Since rewards are terminal, we need to store full episodes.
    """
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self):
        """Return all stored data."""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
        }
    
    def generate_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'batches': batches,
        }

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


# ============================================================================
# PPO Agent
# ============================================================================
class PPO_Agent:
    def __init__(
        self,
        policy: ActorCritic,
        gamma: float = 1.0,  # No discounting for terminal-only rewards
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.policy = policy
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation.
        
        For terminal-only rewards in curling:
        - All intermediate rewards are 0
        - Final reward is win/loss/tie outcome
        - gamma=1.0 makes sense (no time preference)
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        for t in range(len(rewards)-1):
            discount = 1
            advantage = 0
            for k in range(t, len(rewards)-1):
                advantage += discount * rewards[k] + self.gamma * values[k+1] * (1 - int(dones[k])) - values[k]
                discount *= self.gamma * self.gae_lambda
            advantages[t] = advantage
        return advantages
    
    def update(self, trajectory_buffer: TrajectoryBuffer):
        """
        Update policy using trajectories from both teams.
        
        Both teams' experiences are valid training data for the
        same network, just from different perspectives (indicated by team_id).
        """

        stats = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}
        
        for _ in range(self.ppo_epochs):
            # Get data from both buffers
            data = trajectory_buffer.generate_batch()
            
            advantages = self.compute_gae(data['rewards'], data['values'], data['dones'])

            # Normalize advantages (across both teams)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Mini-batch updates
            for batch in data['batches']:
                
                # Convert to tensors
                batch_states = torch.tensor(data['states'][batch]).to(self.policy.device)
                batch_bin_indices = torch.tensor(data['actions'][batch]).to(self.policy.device)
                batch_values = torch.tensor(data['values'][batch]).to(self.policy.device)
                batch_old_log_probs = torch.tensor(data['log_probs'][batch]).to(self.policy.device)
                batch_advantages = torch.tensor(advantages[batch]).to(self.policy.device)
                
                # Forward pass
                dists, values = self.policy(batch_states)

                new_log_probs, entropy = self.policy.get_log_prob(dists, batch_bin_indices)

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                weighted_probs = ratio * batch_advantages
                weighted_clipped_probs = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                batch_returns = batch_advantages + batch_values
                policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Value loss
                value_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                # Track statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    stats['policy_loss'].append(policy_loss.item())
                    stats['value_loss'].append(value_loss.item())
                    stats['entropy'].append(entropy.item())
                    stats['approx_kl'].append(approx_kl)

        trajectory_buffer.clear()

        # Return averaged statistics
        return {k: np.mean(v) for k, v in stats.items()}
    
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)


if __name__ == "__main__":
    # Test ActorCritic instantiation
    STATE_DIM = 3 * 2 * 5 + 1  # Example for 5 stones per team
    ACTION_DIM = 3
    HIDDEN_DIM = 256
    action_bin = (10, 10, 10)
    model = ActorCritic(STATE_DIM, ACTION_DIM, alpha=3e-4, hidden_size=HIDDEN_DIM, action_bin=action_bin)
    print(model)

    # Test forward pass with dummy input
    dummy_state = torch.randn(2, STATE_DIM).to(model.device)
    dists, value = model(dummy_state)
    action, bin_indices = model.get_action(dists)
    log_prob, entropy = model.get_log_prob(dists, bin_indices)
    print("Dists:", dists)
    print("Action:", action)
    print("Bin Indices:", bin_indices)
    print("Value:", value)
    print("Log Prob:", log_prob)
    print("Entropy:", entropy)
