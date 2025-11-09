from curling_env import CurlingEnv
from ppo_agent import PPO_Agent, TrajectoryBuffer, ActorCritic
import torch
import matplotlib.pyplot as plt


# ============================================================================
# Self-Play Collector
# ============================================================================


def collect_self_play_trajectory(env, policy: ActorCritic, buffer: TrajectoryBuffer, device='cpu'):
    state, info = env.reset()
    # print(len(state))
    done = False
    current_team = env.current_team
    
    step_count = 0
    
    while not done:
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Get action from policy
        dist, value = policy(state)
        action, log_prob, entropy = policy.get_action(dist)

        # print(action[0])

        # Step environment
        next_state, reward, done, info = env.step(action[0])

        # Store in appropriate buffer
        # Note: Intermediate rewards are 0, only terminal reward matters
        state = state.detach().numpy()
        action = action.detach().numpy()
        log_prob = log_prob.detach().numpy()
        value = value.detach().numpy()
        
        if current_team == 0:
            buffer.add(state, action[0], log_prob, value, reward[0], done)
        else:
            buffer.add(state, action[0], log_prob, value, reward[1], done)

        state = next_state
        current_team = env.current_team
        step_count += 1

    game_info = {
        'steps': step_count,
        'team1_reward': reward[0],
        'team2_reward': reward[1],
        'team1_score': info['score_team_a'],
        'team2_score': info['score_team_b'],
    }

    return buffer, game_info


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
    LEARNING_RATE = 3e-4
    
    # PPO config
    GAMMA = 1.0  # No discounting for terminal rewards
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    PPO_EPOCHS = 4
    BATCH_SIZE = 64

    n_games = 500

    file_path = f"models/ppo_agent_{n_games}.pth"

    env = CurlingEnv(stones_per_team=stones_per_team, render_mode=None)

    # Initialize policy and trainer
    policy = ActorCritic(STATE_DIM, ACTION_DIM, alpha=LEARNING_RATE, hidden_size=HIDDEN_DIM)
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
    N = 10

    team_1_rewards = []
    team_2_rewards = []
    team_1_scores = []
    team_2_scores = []

    for game in range(1, n_games + 1):
        print(f"Game {game}/{n_games}")
        trajectory_buffer, game_info = collect_self_play_trajectory(env, policy, trajectory_buffer, device=device)

        # Log game info
        print(f"Game {game} - Steps: {game_info['steps']}, Team 1 Reward: {game_info['team1_reward']}, Team 2 Reward: {game_info['team2_reward']}")
        team_1_rewards.append(game_info['team1_reward'])
        team_2_rewards.append(game_info['team2_reward'])
        team_1_scores.append(game_info['team1_score'])
        team_2_scores.append(game_info['team2_score'])
        if game % N == 0:
            trainer.update(trajectory_buffer)
            learn_iters += 1
            trajectory_buffer.clear()
            print(f"Learning iteration {learn_iters} completed.")

    # Save the trained model
    trainer.save(file_path)
    print(f"Trained model saved to {file_path}")

    # Plotting the rewards
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_games + 1), team_1_rewards, label='Team 1 Rewards')
    plt.plot(range(1, n_games + 1), team_2_rewards, label='Team 2 Rewards')
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.title('Self-Play Rewards Over Time')
    plt.legend()
    plt.savefig(f"ppo_rewards_{n_games}.png")
    plt.show()

    # Plotting the scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_games + 1), team_1_scores, label='Team 1 Scores')
    plt.plot(range(1, n_games + 1), team_2_scores, label='Team 2 Scores')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.title('Self-Play Scores Over Time')
    plt.legend()
    plt.savefig(f"ppo_scores_{n_games}.png")
    plt.show()