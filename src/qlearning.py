import numpy as np
import gym
from collections import deque


class QLearning:
    def __init__(self, n_states, n_actions, gamma, alpha=0.01, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                 initialization_method="optimistic", optimistic_initializer=1.0):
        self.nS = n_states
        self.nA = n_actions
        self.Q_table = self.initialize_Q(initialization_method, optimistic_initializer)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def initialize_Q(self, method, optimistic_initializer):
        if method is "random":
            return np.random.random((self.nS, self.nA))
        if method is "optimistic":
            return np.full((self.nS, self.nA), optimistic_initializer, dtype=np.float64)

    def learn(self, current_state, current_action, reward, next_state, done):
        current_value = self.Q_table[current_state][current_action]
        target = self.gamma * np.max(self.Q_table[next_state]) if not done else 0
        delta_t = reward + target - current_value

        self.Q_table[current_state][current_action] = current_value + self.alpha * delta_t
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def choose_action(self, next_state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.nA)
        return np.argmax(self.Q_table[next_state])


if __name__ == "__main__":
    n_episodes = 5000
    env = gym.make("FrozenLake-v0", map_name="4x4")

    n_states = env.nS
    n_actions = env.nA

    sarsa = QLearning(n_states, n_actions, gamma=1)
    scores = deque(maxlen=100)

    for i in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = sarsa.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            sarsa.learn(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        scores.append(steps)
        if i % 100 == 0:
            print(f"Episode {i}, Mean Score {np.mean(scores)}")