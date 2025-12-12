from kaggle_environments import make, evaluate, utils, agent
import inspect
import sys
from IPython.display import clear_output


def my_agent(obs, config):
    import random
    import numpy as np

    N_STEPS = 3  # Number of steps to look ahead

    # Calculates score if agent drops piece in selected column, considering multiple steps ahead
    def score_move(grid, col, mark, config, depth, is_maximizing_player):
        next_grid = drop_piece(grid, col, mark, config)
        if depth == 0 or is_terminal(next_grid, config):
            return get_heuristic(next_grid, mark, config)

        # Simulate opponent's move by alternating player
        next_mark = mark % 2 + 1
        if is_maximizing_player:
            return minimax(next_grid, next_mark, config, depth - 1, False)
        else:
            return minimax(next_grid, next_mark, config, depth - 1, True)

    # Minimax function to recursively search moves
    def minimax(grid, mark, config, depth, is_maximizing_player):
        valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
        if is_maximizing_player:
            max_eval = -float('inf')
            for col in valid_moves:
                eval = score_move(grid, col, mark, config, depth, True)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for col in valid_moves:
                eval = score_move(grid, col, mark, config, depth, False)
                min_eval = min(min_eval, eval)
            return min_eval

    # Helper function to check if a terminal state is reached (win/loss/draw)
    def is_terminal(grid, config):
        return check_victory(grid, 1, config) or check_victory(grid, 2, config) or len(
            [c for c in range(config.columns) if grid[0][c] == 0]) == 0

    # Helper function to check if a player has won
    def check_victory(grid, mark, config):
        return count_windows(grid, 4, mark, config) > 0

    # Helper function for score_move: gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for score_move: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
        score = num_threes - 1e2 * num_threes_opp + 1e6 * num_fours
        return score

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use minimax to assign a score to each possible board in the next N_STEPS turns
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS - 1, True) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)


def write_agent_to_file(function, file):
    with open(file, "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)


env = make("connectx", debug=True)
config = env.configuration
ROWS = 6
COLUMNS = 7
config.columns = COLUMNS
config.rows = ROWS
config.inarow = 4

print(env.name, env.version)
# List of available default agents
print("Default Agents: ", list(env.agents))

env.run([my_agent, "negamax"])
# env.run([agent1, agent2])


print()

output = env.render(mode="ansi")
print(output)

htmloutput = env.render(mode="html")
print(htmloutput)

write_agent_to_file(my_agent, "submission.py")

# Note: Stdout replacement is a temporary workaround.
out = sys.stdout
submission = utils.read_file("submission.py")
agent = agent.get_last_callable(submission, path=submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
