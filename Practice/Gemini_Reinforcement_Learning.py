from kaggle_environments import make, evaluate, utils, agent
import inspect
import sys
from IPython.display import clear_output
import numpy as np
import random
from tensorflow.keras.layers import Dense, Input
from keras import Model
from collections import deque

# Create the ConnectX environment
env = make("connectx", debug=True)
config = env.configuration


# Define the agent
def my_agent(obs, config):
    import numpy as np
    import random
    from tensorflow.keras.layers import Dense, Input
    from keras import Model
    from collections import deque

    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    # Define the neural network model
    def create_model(config):
        input_layer = Input(shape=(config.rows * config.columns,))
        hidden_layer1 = Dense(128, activation='relu')(input_layer)
        hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
        output_layer = Dense(config.columns, activation='linear')(hidden_layer2)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    # Helper function to drop a piece onto the grid
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function to calculate the reward
    def get_reward(grid, mark, config):
        # Check for win
        if is_terminal_node(grid, mark, config):
            return 1
        # Check for draw
        elif list(grid[0, :]).count(0) == 0:
            return 0.5
        else:
            return 0

    # Helper function to check if the game has ended
    def is_terminal_node(grid, mark, config):
        # Check for win: horizontal, vertical, or diagonal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if window.count(mark) == config.inarow:
                    return True
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if window.count(mark) == config.inarow:
                    return True
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if window.count(mark) == config.inarow:
                    return True
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if window.count(mark) == config.inarow:
                    return True
        return False

    # Initialize the model
    model = create_model(config)

    # Define the replay buffer
    replay_buffer = deque(maxlen=2000)

    # Define the exploration rate
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # Define the discount factor
    gamma = 0.95

    # Choose an action using the epsilon-greedy policy
    if np.random.rand() <= epsilon:
        # Explore: choose a random valid action
        valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
        action = random.choice(valid_moves)
    else:
        # Exploit: choose the action with the highest predicted Q-value
        state = np.array(obs.board).reshape(1, -1)
        q_values = model.predict(state)
        valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
        valid_q_values = [q_values[0][c] for c in valid_moves]
        action = valid_moves[np.argmax(valid_q_values)]

    # Store the experience in the replay buffer
    next_state = drop_piece(grid, action, obs.mark, config)
    reward = get_reward(next_state, obs.mark, config)
    replay_buffer.append((grid, action, reward, next_state))

    # Train the model if the replay buffer has enough samples
    if len(replay_buffer) > 32:
        batch = random.sample(replay_buffer, 32)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states).reshape(32, -1)
        next_states = np.array(next_states).reshape(32, -1)
        targets = model.predict(states)
        q_futures = model.predict(next_states)
        for i in range(32):
            targets[i, actions[i]] = rewards[i] + gamma * np.max(q_futures[i])
        model.train_on_batch(states, targets)

    # Decay the exploration rate
    epsilon *= epsilon_decay
    epsilon = max(epsilon, epsilon_min)

    return int(action)

# Write the agent to a file
def write_agent_to_file(function, file):
    with open(file, "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)


env = make("connectx", debug=True)

# Initialize agent
# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

config = env.configuration
ROWS = 6
COLUMNS = 7
config.columns = COLUMNS
config.rows = ROWS
config.inarow = 4

print(env.name, env.version)
# List of available default agents
print("Default Agents: ", list(env.agents))

# env.run(["random", "random"])
env.run([my_agent, my_agent])

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
