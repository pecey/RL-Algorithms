import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from _collections import deque
import numpy as np


class DQN():
    def __init__(self, n_actions, n_features, memory_length, learning_rate):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(n_features, ), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(n_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))

        self.memory = deque(maxlen = memory_length)

    def next_action(self, epsilon, current_state):
        if np.random.random() < epsilon:
            return np.random.randint(low = 0, high = n_actions)
        q_values = self.model.predict(current_state.reshape(1, -1))
        return np.argmax(q_values[0])

    def experience_replay(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size)
        batch = np.array(self.memory)[batch_indexes]
        for state, action, reward, next_state, done in batch:
            if done:
                qsa = reward
            else:
                qsa = reward + gamma * np.max(self.model.predict(next_state)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = qsa
            self.model.fit(state, q_values, verbose = False)


    def add_to_experience(self, experience):
        self.memory.append(experience)


def evaluate(env, dqn):
    n_iterations = 100
    rewards = np.array([])
    for iteration in range(n_iterations):
        state = env.reset().reshape(1, -1)
        done = False
        iteration_reward = 0
        while not done:
            action = dqn.next_action(epsilon, state)
            next_state, reward, done, info = env.step(action)
            iteration_reward += reward
            state = next_state
        rewards = np.append(rewards, iteration_reward)
    return np.mean(rewards)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    feature_space_shape = env.observation_space.shape
    n_actions = env.action_space.n

    learning_rate = 0.01
    epsilon = 0.4
    episodes_to_train = 1000
    dqn = DQN(n_actions, feature_space_shape[0], 1000, learning_rate)

    for episode in range(episodes_to_train):
        state = env.reset().reshape(1, -1)
        steps = 0
        done = False
        while not done:
            steps += 1
            action = dqn.next_action(epsilon, state)
            next_state, reward, done, info = env.step(action)
            dqn.add_to_experience((state.reshape(1, -1), action, reward, next_state.reshape(1,-1), done))
            state = next_state
            dqn.experience_replay(20, 0.99)
        if episode % 100 == 0:
            print(evaluate(env, dqn))



