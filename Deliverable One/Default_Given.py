from kaggle_environments import make, evaluate, utils, agent
import inspect
import sys
from IPython.display import clear_output


def my_agent(obs, config):
    import random
    import numpy as np

    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = piece
        return next_grid

    # Returns True if dropping piece in column results in game win
    def check_winning_move(obs, config, col, piece):
        # Convert the board to a 2D grid
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        next_grid = drop_piece(grid, col, piece, config)
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[row, col:col + config.inarow])
                if window.count(piece) == config.inarow:
                    return True
        # vertical
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(next_grid[row:row + config.inarow, col])
                if window.count(piece) == config.inarow:
                    return True
        # positive diagonal
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if window.count(piece) == config.inarow:
                    return True
        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if window.count(piece) == config.inarow:
                    return True
        return False

    board = obs.board
    columns = config.columns
    return [c for c in range(columns) if board[c] == 0][0]

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

env.run(["random", "random"])
# env.run([agent1, agent2])


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
