import numpy as np
import gym
from copy import deepcopy

    
class Node():
    """
    Attributes:
        state: the state value
        n_visits: number of visits made to the state
        children: nodes containing the next states
    """

    def __init__(self, state, value=0, parent=None):
        self.state = state
        self.value = value
        self.n_visits = 0
        self.parent = parent
        self.children = dict()

    def add_child(self, action, next_state):
        self.children[action] = next_state

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


class MCTS():
    def __init__(self, env):
        self.env = env
        self.policy = {}
        self.n_rollouts = 10
        self.rollout_depth = 20

    """
    Select the node with the maximum upper confidence bound
    """

    def __selection(self, node, timestep):
        next_node_index = np.argmax([self._upper_confidence_bound(child, timestep) for child in node.children.values()])
        return node.children[next_node_index]

    def _upper_confidence_bound(self, node, timestep):
        return node.value + 2 * np.sqrt(np.log(timestep) / node.n_visits)

    """
    Randomly choose from one of the available actions
    """

    def __expansion(self):
        random_action = np.random.choice(self.env.action_space.n)
        next_state, reward, done, _ = self.env.step(random_action)
        return random_action, next_state

    """
    Performs multiple rollouts from the current state to get value function estimate
    """

    def __rollout(self):
        return np.mean([self.__single_rollout() for idx in range(self.n_rollouts)])

    def __single_rollout(self):
        temp_env = deepcopy(self.env)
        done = False
        total_reward = 0
        while not done:
            random_action = np.random.choice(temp_env.action_space.n)
            _, reward, done, _ = temp_env.step(random_action)
            total_reward += reward
        return total_reward

    """
    Learn the traversal policy
    """

    def learn(self):
        current_state = self.env.reset()
        current_state_node = Node(current_state)
        timestep = 0
        while True:
            if current_state_node.is_leaf():
                action_chosen, next_state = self.__expansion()
                value = self.__rollout()
                next_state_node = Node(next_state, value=value, parent=current_state_node)
                current_state_node.add_child(action_chosen, next_state_node)
                self.backprop(current_state_node, value)
            else:
                next_state_node = self.__selection(current_state_node, timestep=timestep)

            timestep += 1

    """
    Backpropagate the estimated values
    """

    def backprop(self, node, value):
        while node:
            node.value += value
            node = node.parent


def main():
    env = gym.make('CartPole-v1')
    mcts = MCTS(env)
    mcts.learn()


if __name__ == "__main__":
    main()