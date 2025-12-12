import gym
import numpy as np
from math import exp, log
import random
from random import choice, uniform
from collections import deque
import keras
from keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from kaggle_environments import evaluate, make
import matplotlib.pyplot as plt


class ConnectX(gym.Env):

    def __init__(self, switch_prob=0.5):
        self.env = make('connectx', debug=True)
        self.pair = [None, 'random']
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(config.rows, config.columns, 1), dtype=int)

    def switch_side(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def switch_trainer(self):
        current_trainer_random = 'random' in self.pair
        if current_trainer_random:
            self.pair = [None, 'negamax']
        else:
            self.pair = [None, 'random']
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        return self.trainer.step(action)

    def reset(self):
        if random.uniform(0, 1) < self.switch_prob:  # switch side
            self.switch_side()
        # if random.uniform(0, 1) < self.switch_prob: # switch trainer
        #    self.switch_trainer()
        return self.trainer.reset()


# Deep Q-learning Agent
class DQNAgent:

    def __init__(self, state_size, action_size, episodes):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.gamma = 0.9  # discount rate
        self.epsilon = 0.10  # initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exp((log(self.epsilon_min) - log(self.epsilon)) / (
                    0.8 * episodes))  # reaches epsilon_min after 80% of iterations
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=1e-5))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Exploration
            return choice([c for c in range(self.action_size) if state[:, c] == 0])
            # when exploring, I allow for "wrong" moves to give the agent a chance
            # to experience the penalty of choosing full columns
            # return choice([c for c in range(self.action_size)])
        act_values = self.model.predict(state)  # Exploitation
        action = np.argmax(act_values[0])
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# initialize gym environment and the agent
env = ConnectX(switch_prob=0.5)
state_size = env.observation_space.shape[1]*env.observation_space.shape[0]
action_size = env.observation_space.shape[1]
episodes = 40000
agent = DQNAgent(state_size, action_size, episodes)
agent.load("C:/Users/Dariu/PycharmProjects/AI/Group Project/Deliverable Two/connectX-weights_deep.h5") # load prelearned weights
batch_size = 40 # Don't know if this number makes sense

# Monitoring devices
all_total_rewards = np.empty(episodes)
all_avg_rewards = np.empty(episodes)

# Iterate the game
for e in range(episodes):
    # reset state in the beginning of each game
    done = False
    state = env.reset()
    total_rewards = 0
    while not done:
        # Decide action
        action = int(agent.act(np.array([state.board])))
        next_state, reward, done, _ = env.step(action)
        if not done:
            reward = 0.0/42 # default: reward of 0.5 if not done/ 1 if win/ 0 if lost
        if done:
            if reward == 1: # Won
                reward = 1
            elif reward == 0: # Lost
                reward = -1
            else: # Draw
                reward = 0
        if state.board[action]!=0: # invalid move: hard penalization
            reward = -10
        agent.memorize(np.array([state.board]), action, reward, np.array([next_state.board]), done)
        # make next_state the new current state for the next frame.
        state = next_state
        total_rewards += reward
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        all_total_rewards[e] = total_rewards
        avg_reward = all_total_rewards[max(0, e - 100):e].mean()
        all_avg_rewards[e] = avg_reward
        if e % 100 == 0 :
            agent.save("./connectX-weights_deep.h5")
            print("episode: {}/{}, epsilon: {:.2f}, average: {:.2f}".format(e, episodes, agent.epsilon, avg_reward))

plt.plot(all_avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Avg rewards (100)')
plt.show()

model = keras.Sequential()()
model.add(Dense(20, input_dim=state_size, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.load_weights('connectX-weights_deep.h5')

layers = []

# Get all layers' weights
for i in range(3):
    weights, biases = model.layers[i].get_weights()
    layers.extend([weights, biases])

fc_layers = list(map(
    lambda x: str(list(np.round(x, 8))) \
        .replace('array(', '').replace(')', '') \
        .replace(' ', '') \
        .replace('\n', '') \
        .replace(',dtype=float32',''),
    layers
))
fc_layers = np.reshape(fc_layers, (-1, 2))

# Create the agent
my_agent = '''def my_agent(observation, configuration):
    import numpy as np

'''
# Write hidden layers
for i, (w, b) in enumerate(fc_layers[:-1]):
    my_agent += '    hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)
    my_agent += '    hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)

my_agent += '    ol_w = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][0])
my_agent += '    ol_b = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][1])
my_agent += '''
    state = observation.board[:]
#    state.append(observation.mark)
    out = np.array(state, dtype=np.float32)
'''

for i in range(len(fc_layers[:-1])):
    my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+1)
    my_agent += '    out = 1/(1 + np.exp(-out))\n' # Sigmoid function

my_agent += '    out = np.matmul(out, ol_w) + ol_b\n'
my_agent += '''
    for i in range(configuration.columns):
        if observation.board[i] != 0:
            out[i] = -1e7

    return int(np.argmax(out))
    '''

with open('../Deliverable Two/submission.py', 'w') as f:
    f.write(my_agent)
