import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional

# Import the physics simulation components
from curlingsim import (
    Stone, simulate
)


class CurlingEnv(gym.Env):
    """
    OpenAI Gym environment for curling with self-play support.
    
    Two teams alternate throwing stones. The environment tracks all stones
    on the sheet and simulates physics including collisions.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        stones_per_team: int = 8,
        sheet_length: float = 45.72,  # meters (standard curling sheet)
        sheet_width: float = 4.75,      # meters
        house_center: Tuple[float, float] = (0.0, 34.75),  # center of target
        house_radius: float = 1.83,    # meters (12 foot diameter)
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.stones_per_team = stones_per_team
        self.total_stones = stones_per_team * 2
        self.sheet_length = sheet_length
        self.sheet_width = sheet_width
        self.house_center = np.array(house_center)
        self.house_radius = house_radius
        self.render_mode = render_mode
        
        # Game state
        self.team_a_stones: List[Stone] = []
        self.team_b_stones: List[Stone] = []
        self.current_team = 0  # 0 for team A, 1 for team B
        self.stones_thrown = 0
        self.done = False
        
        # Starting position for throwing stones
        self.throw_line_x = 0.0
        self.throw_line_y = 1.37
        
        # Action space: [speed (m/s), angle (radians), spin (rad/s)]
        self.action_space = spaces.Box(
            low=np.array([0.5, 70, -10.0]),
            high=np.array([4.0, 110, 10.0]),
            dtype=np.float32
        )
        
        # Observation space: positions of all stones + game state
        # For each stone: [x, y, team_id]
        # Plus: [current_team]
        obs_dim = self.total_stones * 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.team_a_stones = []
        self.team_b_stones = []
        self.current_team = 0
        self.stones_thrown = 0
        self.done = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one stone throw.
        
        Args:
            action: [speed, angle, spin] for the stone throw
            
        Returns:
            observation, reward, terminated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        
        # Extract action parameters
        speed = float(action[0])
        angle = math.radians(float(action[1]))  # psi: heading angle
        spin = float(action[2])   # omega: angular velocity (positive = CW)
        
        # Create the new stone at the throw line
        new_stone = Stone(
            x=self.throw_line_x,
            y=self.throw_line_y,
            v=speed,
            psi=angle,
            omega=spin,
            team=self.current_team,
            phi=0.0,
            moving=True
        )
        
        # Get all existing stones for simulation
        all_stones = self.team_a_stones + self.team_b_stones + [new_stone]
        
        # Simulate the throw
        simulate(all_stones, dt=0.001, t_max=30.0)
        
        # Add the new stone to the appropriate team
        if self.current_team == 0:
            self.team_a_stones.append(new_stone)
        else:
            self.team_b_stones.append(new_stone)
        
        # Update game state
        self.stones_thrown += 1
        
        # Check if game is over
        terminated = (self.stones_thrown >= self.total_stones)
        self.done = terminated
        
        # Switch teams for next throw
        if not terminated:
            self.current_team = 1 - self.current_team
            reward = (0, 0)
        else:
            # Calculate reward (score difference for current team)
            # reward = self._calculate_reward()
            reward = self.game_results()

        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Create observation vector containing all stone states.
        """
        obs = []

        # Add all stones (team A first, then team B)
        for i in range(self.stones_per_team):
            if i < len(self.team_a_stones):
                stone = self.team_a_stones[i]
                obs.extend([stone.x, stone.y, float(stone.team)])
            else:
                # Placeholder for stones not yet thrown
                obs.extend([0.0, 0.0, 0.0])
        for i in range(self.stones_per_team):
            if i < len(self.team_b_stones):
                stone = self.team_b_stones[i]
                obs.extend([stone.x, stone.y, float(stone.team)])
            else:
                # Placeholder for stones not yet thrown
                obs.extend([0.0, 0.0, 1.0])

        # Add game state info
        obs.append(float(self.current_team))

        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Return auxiliary information about the environment state."""
        return {
            'current_team': self.current_team,
            'stones_thrown': self.stones_thrown,
            'team_a_count': len(self.team_a_stones),
            'team_b_count': len(self.team_b_stones),
            'score_team_a': self._score_team(0),
            'score_team_b': self._score_team(1),
        }
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current stone positions.
        
        For the current team that just threw, reward is:
        (score_current_team - score_opponent_team)
        
        This encourages getting closer to the house center than opponent.

        Returns a tuple of score difference (team_a, team_b).
        """
        score_a = self._score_team(0)
        score_b = self._score_team(1)
        
        return (float(score_a - score_b), float(score_b - score_a))
    
    def _score_team(self, team: int) -> float:
        """
        Calculate score for a team based on distance to house center.
        
        In curling, only stones closer than the opponent's closest stone count.
        For RL training, we use a continuous reward based on inverse distance.
        """
        if team == 0:
            team_stones = self.team_a_stones
        else:
            team_stones = self.team_b_stones
        
        if len(team_stones) == 0:
            return 0.0
        
        # Calculate distances to house center for this team
        distances = []
        for stone in team_stones:
            dx = stone.x - self.house_center[0]
            dy = stone.y - self.house_center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Only count stones within the house
            # if dist <= self.house_radius:
            distances.append(dist)
        
        if len(distances) == 0:
            return 0.0
        
        # Simple scoring: inverse of minimum distance (closer is better)
        min_dist = min(distances)
        return 1.0 / (min_dist + 0.1)  # +0.1 to avoid division by zero
    
    def game_results(self) -> Dict[str, float]:
        """
        Return final scores for both teams at the end of the game.
        Sort the stones for both teams by distance to house center.
        If the closest n stones belong to one team, that team scores n points.
        0 points if the closest stone is from the opponent.
        """
        all_stones = self.team_a_stones + self.team_b_stones
        if len(all_stones) == 0:
            return {
                'team_a_score': 0.0,
                'team_b_score': 0.0,
            }
        # Calculate distances to house center
        stone_distances = []
        for stone in all_stones:
            dx = stone.x - self.house_center[0]
            dy = stone.y - self.house_center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            stone_distances.append((dist, stone.team))
        # Sort by distance
        stone_distances.sort(key=lambda x: x[0])
        # Determine scoring
        scoring_team = stone_distances[0][1]
        team_a_score = 0.0
        team_b_score = 0.0
        for dist, team in stone_distances:
            if team == scoring_team:
                if team == 0:
                    team_a_score += 1.0
                else:
                    team_b_score += 1.0
            else:
                break  # Stop counting when opponent's stone is reached

        return (team_a_score, team_b_score)

    def get_stone_positions(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Get current positions of all stones for both teams.
        
        Returns:
            (team_a_positions, team_b_positions) as lists of (x, y) tuples
        """
        team_a_pos = [(s.x, s.y) for s in self.team_a_stones]
        team_b_pos = [(s.x, s.y) for s in self.team_b_stones]
        return team_a_pos, team_b_pos
    
    def render(self):
        """Render the current state of the curling sheet."""
        if self.render_mode is None:
            return
        
        if self.render_mode == "human":
            # Simple text-based rendering
            print(f"\n{'='*50}")
            print(f"Curling Game - Stones Thrown: {self.stones_thrown}/{self.total_stones}")
            print(f"Current Team: {'A' if self.current_team == 0 else 'B'}")
            print(f"{'='*50}")
            
            print(f"\nTeam A stones ({len(self.team_a_stones)}):")
            for i, stone in enumerate(self.team_a_stones):
                dist = math.sqrt(
                    (stone.x - self.house_center[0])**2 + 
                    (stone.y - self.house_center[1])**2
                )
                print(f"  {i+1}. Position: ({stone.x:.2f}, {stone.y:.2f}), "
                      f"Distance to house: {dist:.2f}m")
            
            print(f"\nTeam B stones ({len(self.team_b_stones)}):")
            for i, stone in enumerate(self.team_b_stones):
                dist = math.sqrt(
                    (stone.x - self.house_center[0])**2 + 
                    (stone.y - self.house_center[1])**2
                )
                print(f"  {i+1}. Position: ({stone.x:.2f}, {stone.y:.2f}), "
                      f"Distance to house: {dist:.2f}m")
            
            scores = self._get_info()
            print(f"\nScores - Team A: {scores['score_team_a']:.3f}, "
                  f"Team B: {scores['score_team_b']:.3f}")
            print(f"{'='*50}\n")


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = CurlingEnv(stones_per_team=5, render_mode="human")
    
    print("Testing Curling Environment")
    print("="*50)
    
    # Reset environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    actions = [(3, 100, 2), (3, 95, 2), (3, 96, 2), (2.8, 96, 2), (2.9, 96, 2), (3, 96, 2)]

    # Play a few random actions
    for i in range(6):
        # Sample random action
        action = actions[i]
        print(f"\nThrow {i+1}: speed={action[0]:.2f}, angle={action[1]:.3f}, spin={action[2]:.2f}")
        
        obs, reward, terminated, info = env.step(action)
        
        env.render()
        print(f"Reward: {reward}")
        # print(obs)
        if terminated:
            print("\nGame Over!")
            break
    
    print("\nEnvironment test complete!")