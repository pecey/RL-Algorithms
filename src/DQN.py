import gym
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, learning_rate, input_dims, l1_dims, l2_dims, n_actions):
        super().__init__()
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.n_actions = n_actions
        self.lin1 = nn.Linear(*input_dims, self.l1_dims)
        self.lin2 = nn.Linear(self.l1_dims, self.l2_dims)
        self.lin3 = nn.Linear(self.l2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        actions = self.lin3(x)

        return actions


class Agent():
    def __init__(self, gamma, epsilon, learning_rate, input_dims, batch_size, n_actions,
                 memory_size=1000000, min_epsilon=0.01, epsilon_decay=0.995):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = min_epsilon
        self.eps_dec = epsilon_decay
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = list(range(n_actions))
        self.Q_network = DQN(learning_rate, input_dims=input_dims,
                             n_actions=n_actions,
                             l1_dims=24, l2_dims=24)
        self.memory = deque(maxlen=memory_size)

    def remember_experience(self, experience):
        self.memory.append(experience)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        actions = self.Q_network.forward(observation)
        return torch.argmax(actions).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.Q_network.optimizer.zero_grad()

        batch_indexes = np.random.choice(len(self.memory), self.batch_size)
        replay_batch = np.array(self.memory)[batch_indexes]
        observation_batch, action_batch, reward_batch, next_observation_batch, terminal_batch = list(zip(*replay_batch))

        reward_batch = torch.Tensor(reward_batch).to(self.Q_network.device)
        # False is 0, True is 1
        terminal_batch = torch.Tensor(terminal_batch).type(torch.int8).to(self.Q_network.device)

        q_eval = self.Q_network.forward(observation_batch).to(self.Q_network.device)
        q_target = self.Q_network.forward(observation_batch).to(self.Q_network.device)
        q_next = self.Q_network.forward(next_observation_batch).to(self.Q_network.device)
        q_target[np.arange(self.batch_size), action_batch] = reward_batch + self.gamma * torch.max(q_next, dim=1).values * (1 - terminal_batch)

        loss = self.Q_network.loss(q_target, q_eval).to(self.Q_network.device)
        loss.backward()
        self.Q_network.optimizer.step()

        self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)


if __name__ == "__main__":
    n_episodes = 1000

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    feature_space_shape = env.observation_space.shape[0]

    agent = Agent(gamma=1, epsilon=1.0, input_dims=[feature_space_shape], batch_size=64, n_actions=n_actions, learning_rate=0.01)
    scores = deque(maxlen = 100)
    for i in range(n_episodes):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            #env.render()
            next_observation, reward, done, info = env.step(action)
            agent.remember_experience((observation, action, reward, next_observation, done))
            agent.learn()
            observation = next_observation
            score += reward
        scores.append(score)
        print(f"Episode: {i}, Score: {score}")
        if i % 100 == 0:
            print(f"Mean Score: {np.mean(scores)}, Epsilon: {agent.epsilon}")
