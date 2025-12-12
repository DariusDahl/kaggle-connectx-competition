from kaggle_environments import make, utils, agent
import inspect
import sys


def my_agent(obs, config):
    import random
    import numpy as np

    # Dynamic depth adjustment based on game state
    def get_depth(grid, config):
        empty_cells = np.count_nonzero(grid == 0)
        if empty_cells > 30:
            return 3
        elif empty_cells > 15:
            return 4
        else:
            return 5

    # Calculates the score for a move
    def score_move(grid, col, mark, config, depth, alpha, beta):
        next_grid = drop_piece(grid, col, mark, config)
        score = alpha_beta(next_grid, depth - 1, False, mark, config, alpha, beta)
        return score

    # Drops a piece in a column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                next_grid[row][col] = mark
                break
        return next_grid

    # Evaluates the grid with an enhanced heuristic
    def get_heuristic(grid, mark, config):
        center_score = np.count_nonzero(grid[:, config.columns // 2] == mark) * 3
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
        num_fours_opp = count_windows(grid, 4, mark % 2 + 1, config)
        score = center_score + num_threes - 1e2 * num_threes_opp - 1e4 * num_fours_opp + 1e6 * num_fours
        return score

    # Counts the number of windows satisfying specific conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    # Helper function to check if a window satisfies conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)

    # Alpha-beta pruning implementation
    def alpha_beta(node, depth, maximizingPlayer, mark, config, alpha, beta):
        if depth == 0 or is_terminal_node(node, config):
            return get_heuristic(node, mark, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if maximizingPlayer:
            value = -np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, alpha_beta(child, depth - 1, False, mark, config, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark % 2 + 1, config)
                value = min(value, alpha_beta(child, depth - 1, True, mark, config, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # Checks if the game has ended
    def is_terminal_node(grid, config):
        if list(grid[0, :]).count(0) == 0:
            return True
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False

    # Checks if a window is a terminal window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Get the current board as a 2D array
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    depth = get_depth(grid, config)
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    scores = {col: score_move(grid, col, obs.mark, config, depth, -np.Inf, np.Inf) for col in valid_moves}
    max_score = max(scores.values())
    best_moves = [col for col in valid_moves if scores[col] == max_score]
    return random.choice(best_moves)


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

env.run([my_agent, my_agent])

print()

output = env.render(mode="ansi")
print(output)

# htmloutput = env.render(mode="html")
# print(htmloutput)

write_agent_to_file(my_agent, "submission.py")

# Note: Stdout replacement is a temporary workaround.
out = sys.stdout
submission = utils.read_file("submission.py")
agent = agent.get_last_callable(submission, path=submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
