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
