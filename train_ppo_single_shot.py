from curling_env import CurlingEnv
from ppo_agent_discrete import PPO_Agent, TrajectoryBuffer, ActorCritic
import torch
import numpy as np
from gymnasium.vector import SyncVectorEnv
import matplotlib.pyplot as plt


def make_env(rank, stones_per_team=1):
    """Factory function to create environments"""
    def _make():
        return CurlingEnv(stones_per_team=stones_per_team, render_mode=None)
    return _make


def parallel_collect_trajectories(num_envs, policy, buffer: TrajectoryBuffer, device='cpu'):
    """Collect trajectories from vectorized environments"""
    
    # Create vectorized environment
    env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
    
    states, infos = env.reset()
    game_infos = []
    
    dones = np.zeros(num_envs, dtype=bool)
    step_count = 0
    
    team1_rewards_sum = np.zeros(num_envs)
    team2_rewards_sum = np.zeros(num_envs)

    while not np.all(dones):
        # Batch process all environments
        states_tensor = torch.FloatTensor(states).to(device)
        
        with torch.no_grad():
            dists, values = policy(states_tensor)
            actions, bin_indices = policy.get_action(dists)
            log_probs, _ = policy.get_log_prob(dists, bin_indices)
        
        # Step all environments at once
        next_states, rewards, terminateds, truncateds, infos = env.step(
            actions.detach().cpu().numpy()
        )
        
        dones = terminateds | truncateds
        # print(infos)
        print(infos['current_team'])
        print(rewards)
        # Store trajectories for all active environments
        for i in range(num_envs):
            if not dones[i]:
                buffer.add(
                    states[i:i+1],
                    bin_indices[i:i+1],
                    log_probs[i:i+1],
                    values[i:i+1],
                    rewards[i],
                    dones[i]
                )
                # Accumulate rewards for each team
                # Assume current_team is available in infos[i]
                if infos['current_team'][i] == 0:
                    team1_rewards_sum[i] += rewards[i]
                else:
                    team2_rewards_sum[i] += rewards[i]
            else:
                # Record game info for finished env
                game_info = {
                    'steps': step_count,
                    'team1_result': infos['results_team_a'][i],
                    'team2_result': infos['results_team_b'][i],
                    'team1_reward': team1_rewards_sum[i],
                    'team2_reward': team2_rewards_sum[i],
                }
                game_infos.append(game_info)
        states = next_states
        step_count += 1
    env.close()
    return buffer, game_infos


if __name__ == "__main__":
    # Example usage of the parallel_collect_trajectories function
    num_envs = 4
    stones_per_team = 1
    BATCH_SIZE = num_envs * 128

    # Training config
    LEARNING_RATE = 1e-4
    HIDDEN_DIM = 256

    # PPO config
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    PPO_EPOCHS = 4

    state_dim = CurlingEnv(stones_per_team=stones_per_team).observation_space.shape[0]
    action_dim = 3

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    action_bin = (5, 10, 10)  # Discretization bins for speed, angle, spin
    # Initialize policy and trainer
    policy = ActorCritic(state_dim, action_dim, alpha=LEARNING_RATE, hidden_size=HIDDEN_DIM, action_bin=action_bin)
    print("Initialized policy network.")
    print(policy)
    buffer = TrajectoryBuffer(batch_size=BATCH_SIZE)

    buffer, game_infos = parallel_collect_trajectories(num_envs, policy, buffer, device='cpu')
    
    print("Collected trajectories from vectorized environments.")
    for info in game_infos:
        print(info)