from curling_env import CurlingEnv
from ppo_agent import ActorCritic
import torch


def play_curling(model_1, model_2, env: CurlingEnv, device):
    state, _ = env.reset()
    team1_reward = 0
    team2_reward = 0
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        if env.current_team == 0:  # Team 1's turn
            dist, _ = model_1(state_tensor)
            action = model_1.get_action(dist)
        else:  # Team 2's turn
            dist, _ = model_2(state_tensor)
            action = model_2.get_action(dist)
        next_state, reward, done, info = env.step(action)
        team1_reward += reward[0]
        team2_reward += reward[1]

        env.render()

        state = next_state
    results = env.game_results()
    print("Game over")
    game_info = {
        'team1_result': results[0],
        'team2_result': results[1],
        'team1_reward': team1_reward,
        'team2_reward': team2_reward
    }
    return game_info


# ============================================================================
if __name__ == "__main__":
    stones_per_team = 5
    file_path_1 = "models/ppo_agent_50.pth"
    file_path_2 = "models/ppo_agent_selfplay_5000.pth"

    STATE_DIM = 3 * 2 * stones_per_team + 1
    ACTION_DIM = 3
    HIDDEN_DIM = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_1 = ActorCritic(STATE_DIM, ACTION_DIM, alpha=3e-4, hidden_size=HIDDEN_DIM)
    model_1.load_state_dict(torch.load(file_path_1, map_location=device))
    model_1.to(device)
    model_1.eval()

    model_2 = ActorCritic(STATE_DIM, ACTION_DIM, alpha=3e-4, hidden_size=HIDDEN_DIM)
    model_2.load_state_dict(torch.load(file_path_2, map_location=device))
    model_2.to(device)
    model_2.eval()

    env = CurlingEnv(stones_per_team=stones_per_team, render_mode='human')
    game_info = play_curling(model_1, model_2, env, device)
    print(game_info)
