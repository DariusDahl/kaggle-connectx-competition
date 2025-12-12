from kaggle_environments import make, evaluate, utils, agent
import inspect
import sys
from IPython.display import clear_output
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import gym
import gymnasium as gym
from gym import spaces
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Create the Kaggle environment
ks_env = make("connectx", debug=True)


class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(1,self.rows,self.columns), dtype=int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns)

    def change_reward(self, old_reward, done):
        if old_reward == 1: # The agent won the game
            return 1
        elif done: # The opponent won the game
            return -1
        else: # Reward 1/42
            return 1/(self.rows*self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid: # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns), reward, done, _


# Neural network for predicting action values
class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)


def my_agent(obs, config):
    import numpy as np
    from functools import lru_cache

    # N-step lookahead depth
    N_STEPS = 3

    # Constants for evaluation scoring
    SCORE_WIN = 1e6
    SCORE_OPPONENT_WIN = -1e6
    SCORE_THREE = 50  # Three pieces aligned with a gap
    SCORE_TWO = 10    # Two pieces aligned with a gap
    SCORE_OPPONENT_THREE = -100

    def evaluate_window(window, piece):
        """
        Evaluate a window (of length `config.inarow`) and return a score.
        """
        opp_piece = 3 - piece  # Opponent's piece
        piece_count = window.count(piece)
        opp_count = window.count(opp_piece)

        if piece_count == config.inarow:
            return SCORE_WIN
        elif opp_count == config.inarow:
            return SCORE_OPPONENT_WIN
        elif piece_count == config.inarow - 1 and opp_count == 0:
            return SCORE_THREE
        elif piece_count == config.inarow - 2 and opp_count == 0:
            return SCORE_TWO
        elif opp_count == config.inarow - 1 and piece_count == 0:
            return SCORE_OPPONENT_THREE
        return 0

    def evaluate_board(board, piece):
        """
        Evaluate the entire board, dynamically considering `config.inarow`.
        """
        score = 0
        rows, cols = board.shape

        # Score horizontal
        for row in range(rows):
            for col in range(cols - config.inarow + 1):
                window = list(board[row, col:col + config.inarow])
                score += evaluate_window(window, piece)

        # Score vertical
        for col in range(cols):
            for row in range(rows - config.inarow + 1):
                window = list(board[row:row + config.inarow, col])
                score += evaluate_window(window, piece)

        # Score positive diagonal
        for row in range(rows - config.inarow + 1):
            for col in range(cols - config.inarow + 1):
                window = [board[row + i, col + i] for i in range(config.inarow)]
                score += evaluate_window(window, piece)

        # Score negative diagonal
        for row in range(config.inarow - 1, rows):
            for col in range(cols - config.inarow + 1):
                window = [board[row - i, col + i] for i in range(config.inarow)]
                score += evaluate_window(window, piece)

        return score

    def get_valid_moves(grid, config):
        """Get all valid columns where a piece can be dropped."""
        return [c for c in range(config.columns) if grid[0][c] == 0]

    def drop_piece(grid, col, piece):
        """Simulate dropping a piece in the given column."""
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row, col] == 0:
                next_grid[row, col] = piece
                break
        return next_grid

    @lru_cache(None)
    def minimax(board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning."""
        valid_moves = get_valid_moves(board, config)
        is_terminal = len(valid_moves) == 0 or depth == 0

        if is_terminal:
            if depth == 0:
                return evaluate_board(board, 1)  # Agent's perspective
            else:
                return 0  # No valid moves

        if maximizing_player:
            value = -float('inf')
            for col in valid_moves:
                next_board = drop_piece(board, col, 1)
                value = max(value, minimax(next_board.tobytes(), depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for col in valid_moves:
                next_board = drop_piece(board, col, 2)
                value = min(value, minimax(next_board.tobytes(), depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def select_best_move(board):
        """Select the best move for the agent."""
        valid_moves = get_valid_moves(board, config)
        best_value = -float('inf')
        best_move = random.choice(valid_moves)

        for col in valid_moves:
            next_board = drop_piece(board, col, 1)
            move_value = minimax(next_board.tobytes(), N_STEPS, -float('inf'), float('inf'), False)
            if move_value > best_value:
                best_value = move_value
                best_move = col

        return best_move

    # Convert the flat board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    return select_best_move(grid)


def agent1(obs, config):
    # Use the best model to select a column
    col, _ = model.predict(np.array(obs['board']).reshape(1, 6,7))
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move.
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])


def write_agent_to_file(function, file):
    with open(file, "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)


env = make("connectx", debug=True)

# Initialize agent
# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
# model.learn(total_timesteps=60000)

config = env.configuration
ROWS = 6
COLUMNS = 7
config.columns = COLUMNS
config.rows = ROWS
config.inarow = 4

print(env.name, env.version)
# List of available default agents
print("Default Agents: ", list(env.agents))

env.run([my_agent, my_agent])
# env.run([agent1, agent2])

print()

output = env.render(mode="ansi")
print(output)

# htmloutput = env.render(mode="html")
# print(htmloutput)

write_agent_to_file(my_agent, "../Deliverable Two/submission.py")

# Note: Stdout replacement is a temporary workaround.
out = sys.stdout
submission = utils.read_file("../Deliverable Two/submission.py")
agent = agent.get_last_callable(submission, path=submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
