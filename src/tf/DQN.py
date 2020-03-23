import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from _collections import deque
import numpy as np


class DQN():
    def __init__(self, n_actions, n_features, memory_length, learning_rate, epsilon = 1, epsilon_decay = 0.995, min_epsilon = 0.01):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(n_features, ), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(n_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))

        self.memory = deque(maxlen = memory_length)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def next_action(self, current_state):
        if np.random.random() < self.epsilon:
            return np.random.randint(low = 0, high = n_actions)
        q_values = self.model.predict(current_state)
        return np.argmax(q_values[0])

    def experience_replay(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size)
        batch = np.array(self.memory)[batch_indexes]
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in batch:
            qsa = reward if done else reward + gamma * np.max(self.model.predict(next_state)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = qsa

            x_batch.append(state[0])
            y_batch.append(q_values[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size = len(x_batch), verbose = False)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)


    def add_to_experience(self, experience):
        self.memory.append(experience)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    feature_space_shape = env.observation_space.shape
    n_actions = env.action_space.n

    learning_rate = 0.01
    episodes_to_train = 1000
    dqn = DQN(n_actions, feature_space_shape[0], 1000000, learning_rate)

    scores = deque(maxlen=100)
    for episode in range(episodes_to_train):
        state = env.reset().reshape(1, -1)
        done = False
        score = 0
        while not done:
            action = dqn.next_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(1, -1)
            dqn.add_to_experience((state, action, reward, next_state, done))
            state = next_state
            score += reward
            dqn.experience_replay(64, 1)
        print(f"Episode: {episode}, Score: {score}")
        scores.append(score)
        if episode % 100 == 0:
            print(f"Mean score: {np.mean(scores)}")



