from curling_env import CurlingEnv
from ppo_agent_discrete import PPO_Agent, TrajectoryBuffer, ActorCritic
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
from gymnasium.vector import SyncVectorEnv


# ============================================================================
# Self-Play Collector
# ============================================================================


def collect_self_play_trajectory(env, policy: ActorCritic, buffer: TrajectoryBuffer, device='cpu'):
    state, info = env.reset()
    # print(len(state))
    done = False
    current_team = env.current_team

    step_count = 0

    team_1_rewards = 0
    team_2_rewards = 0
    
    while not done:
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        assert not torch.isnan(state).any(), "State contains NaN"
        assert not torch.isinf(state).any(), "State contains Inf"

        # Get action from policy
        dists, value = policy(state)
        action, bin_indices = policy.get_action(dists)
        log_prob, _ = policy.get_log_prob(dists, bin_indices)

        assert not torch.isnan(action).any(), "Action contains NaN"
        assert not torch.isinf(action).any(), "Action contains Inf"
        assert not torch.isnan(log_prob).any(), "Log prob contains NaN"
        assert not torch.isinf(log_prob).any(), "Log prob contains Inf"
        assert not torch.isnan(value).any(), "Value contains NaN"
        assert not torch.isinf(value).any(), "Value contains Inf"

        # Step environment
        next_state, reward, done, info = env.step(action.detach().cpu().numpy()[0])

        # Store in appropriate buffer
        state = state.detach().numpy()
        action = action.detach().numpy()
        log_prob = log_prob.detach().numpy()
        value = value.detach().numpy()
        buffer.add(state, bin_indices, log_prob, value, reward, done)

        if current_team == 0:
            team_1_rewards += reward
        else:
            team_2_rewards += reward

        state = next_state
        current_team = env.current_team
        step_count += 1

    game_info = {
        'steps': step_count,
        'team1_result': info['results_team_a'],
        'team2_result': info['results_team_b'],
        'team1_reward': team_1_rewards,
        'team2_reward': team_2_rewards,
    }

    return buffer, game_info


def make_env(rank, stones_per_team=5):
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


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":

    stones_per_team = 5
    # Hyperparameters
    STATE_DIM = 3 * 2 * stones_per_team + 1
    ACTION_DIM = 3
    HIDDEN_DIM = 256
    
    # Training config
    LEARNING_RATE = 1e-4
    
    # PPO config
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    PPO_EPOCHS = 4
    BATCH_SIZE = 64

    n_games = 10

    num_envs = 4

    file_path = f"models/ppo_discrete_selfplay_{n_games}_dist.pth"

    # Initialize wandb
    # wandb.init(
    #     project="curling-ppo",
    #     name=f"selfplay_{n_games}_games_dist",
    #     config={
    #         "n_games": n_games,
    #         "stones_per_team": stones_per_team,
    #         "learning_rate": LEARNING_RATE,
    #         "gamma": GAMMA,
    #         "gae_lambda": GAE_LAMBDA,
    #         "clip_epsilon": CLIP_EPSILON,
    #         "value_coef": VALUE_COEF,
    #         "entropy_coef": ENTROPY_COEF,
    #         "ppo_epochs": PPO_EPOCHS,
    #         "batch_size": BATCH_SIZE,
    #         "hidden_dim": HIDDEN_DIM,
    #     }
    # )

    env = CurlingEnv(stones_per_team=stones_per_team, render_mode=None)

    action_bin = (5, 10, 5)  # Discretization bins for speed, angle, spin
    # Initialize policy and trainer
    policy = ActorCritic(STATE_DIM, ACTION_DIM, alpha=LEARNING_RATE, hidden_size=HIDDEN_DIM, action_bin=action_bin)
    trainer = PPO_Agent(
        policy,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    trajectory_buffer = TrajectoryBuffer(BATCH_SIZE)

    learn_iters = 0
    N = 5

    team_1_rewards = []
    team_2_rewards = []
    team_1_results = []
    team_2_results = []

    for game in range(1, n_games + 1):
        print(f"Game {game}/{n_games}")
        # trajectory_buffer, game_info = collect_self_play_trajectory(env, policy, trajectory_buffer, device=device)
        trajectory_buffer, game_infos = parallel_collect_trajectories(num_envs, policy, trajectory_buffer, device=device)

        for game_info in game_infos:
            # Log game info
            print(f"Game {game} - Steps: {game_info['steps']}, Team 1 Reward: {game_info['team1_reward']}, Team 2 Reward: {game_info['team2_reward']}")
            print(f"Results - Team 1: {game_info['team1_result']}, Team 2: {game_info['team2_result']}")

            team_1_rewards.append(game_info['team1_reward'])
            team_2_rewards.append(game_info['team2_reward'])
            team_1_results.append(game_info['team1_result'])
            team_2_results.append(game_info['team2_result'])

            # Log to wandb every game
            # wandb.log({
            #     "game": game,
            #     "team1_reward": game_info['team1_reward'],
            #     "team2_reward": game_info['team2_reward'],
            #     "team1_result": game_info['team1_result'],
            #     "team2_result": game_info['team2_result'],
            # })

        if game % N == 0:
            batch = trajectory_buffer.generate_batch()

            stats = trainer.update(trajectory_buffer)
            print(f"Training stats after iteration {learn_iters}: {stats}")

            # Log training stats to wandb
            # wandb.log({
            #     "learning_iteration": learn_iters,
            #     "training_stats": stats,
            # })

            learn_iters += 1
            trajectory_buffer.clear()
            print(f"Learning iteration {learn_iters} completed.")
        
        if game % 100 == 0:
            policy.eval()
            eval_rewards_1 = []
            eval_rewards_2 = []
            for _ in range(N):
                with torch.no_grad():
                    trajectory_buffer, game_info = collect_self_play_trajectory(env, policy, trajectory_buffer, device=device)
                    eval_rewards_1.append(game_info['team1_reward'])
                    eval_rewards_2.append(game_info['team2_reward'])
            avg_eval_reward_1 = np.mean(np.array(eval_rewards_1))
            avg_eval_reward_2 = np.mean(np.array(eval_rewards_2))
            print(f"Evaluation over {N} games - Team 1 Avg Reward: {avg_eval_reward_1}, Team 2 Avg Reward: {avg_eval_reward_2}")
            # wandb.log({
            #     "eval_team1_avg_reward": avg_eval_reward_1,
            #     "eval_team2_avg_reward": avg_eval_reward_2,
            # })
            trajectory_buffer.clear()
            policy.train()

    # Save the trained model
    trainer.save(file_path)
    print(f"Trained model saved to {file_path}")

    # Plotting the rewards
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_games * num_envs + 1), team_1_rewards, label='Team 1 Rewards')
    plt.plot(range(1, n_games * num_envs + 1), team_2_rewards, label='Team 2 Rewards')
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.title('Self-Play Rewards Over Time')
    plt.legend()
    plt.savefig(f"plots/ppo_discrete_rewards_{n_games}_dist.png")
    plt.show()

    window = 50
    team_1_ma = np.convolve(team_1_rewards, np.ones(window)/window, mode='valid')
    team_2_ma = np.convolve(team_2_rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(window, n_games * num_envs + 1), team_1_ma, label=f'Team 1 Rewards (MA-{window})', linewidth=2)
    plt.plot(range(window, n_games * num_envs + 1), team_2_ma, label=f'Team 2 Rewards (MA-{window})', linewidth=2)
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.title(f'Self-Play Rewards Moving Average (window={window})')
    plt.legend()
    plt.savefig(f"plots/ppo_discrete_rewards_ma_{n_games}_dist.png")
    plt.show()

    # wandb.finish()
