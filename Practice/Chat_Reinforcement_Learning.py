from kaggle_environments import make, evaluate, utils
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from keras import Model
from collections import deque

# Create the ConnectX environment
env = make("connectx", debug=True)
config = env.configuration

# Define the neural network model
def create_model(input_shape, output_shape):
    inputs = Input(shape=(input_shape,))
    hidden1 = Dense(128, activation='relu')(inputs)
    hidden2 = Dense(64, activation='relu')(hidden1)
    outputs = Dense(output_shape, activation='linear')(hidden2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Helper function to drop a piece onto the grid
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            next_grid[row][col] = mark
            break
    return next_grid

# Check if the game has ended and return the reward
def get_reward(grid, mark, config):
    if is_terminal_node(grid, mark, config):
        return 1  # Winning move
    if is_terminal_node(grid, mark % 2 + 1, config):
        return -1  # Opponent winning move
    if list(grid[0, :]).count(0) == 0:
        return 0  # Draw
    return 1/42  # Valid move bonus

# Check if the game is over (win condition)
def is_terminal_node(grid, mark, config):
    # Horizontal check
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if window.count(mark) == config.inarow:
                return True
    # Vertical check
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if window.count(mark) == config.inarow:
                return True
    # Diagonal checks
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

# Deep Q-Learning Agent
def my_agent(obs, config):
    epsilon = 0.1
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.95
    max_memory = 2000
    batch_size = 64

    model = create_model(config.rows * config.columns, config.columns)
    replay_buffer = deque(maxlen=max_memory)

    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    # Epsilon-greedy action selection
    def select_action(state, epsilon):
        if np.random.rand() <= epsilon:
            valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
            return random.choice(valid_moves)
        else:
            state = np.array(state).reshape(1, -1)
            q_values = model.predict(state)
            valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
            q_values = [q_values[0][c] for c in valid_moves]
            return valid_moves[np.argmax(q_values)]

    action = select_action(obs.board, epsilon)

    # Execute the action and observe the reward
    next_state = drop_piece(grid, action, obs.mark, config)
    reward = get_reward(next_state, obs.mark, config)

    # Store the experience in the replay buffer
    replay_buffer.append((grid, action, reward, next_state))

    # Train the model using experience replay
    if len(replay_buffer) > batch_size:
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)

        targets = model.predict(states)
        q_futures = model.predict(next_states)

        for i in range(batch_size):
            targets[i, actions[i]] = rewards[i] + gamma * np.max(q_futures[i])

        model.train_on_batch(states, targets)

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return int(action)

# Write the agent to a file
def write_agent_to_file(function, file):
    import inspect
    with open(file, "w") as f:
        f.write(inspect.getsource(function))


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
