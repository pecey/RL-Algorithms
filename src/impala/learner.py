import torch as T
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from impala.evaluator import evaluator
from collections import deque

from impala.vtrace import vtrace_target

LEARNER_BATCH_SIZE = 4
GAMMA = 0.99
BASELINE_COST = 0.5
ENTROPY_COST = 0.45
LEARNING_RATE = 0.004


def learner(learner_model, queue, parameter_server, lr=0.01):
    scores = deque(maxlen=100)
    """Learner to get trajectories from Actors."""
    optimizer = optim.Adam(learner_model.parameters(), lr=LEARNING_RATE, weight_decay=0.99)
    iteration = 0
    parameter_server.push(learner_model.state_dict())
    while True:
        iteration += 1
        batch_of_trajectories = []
        while len(batch_of_trajectories) < LEARNER_BATCH_SIZE:
            if not queue.empty():
                trajectory = queue.get()
                batch_of_trajectories.append(trajectory)
        states, actions_taken, rewards, dones, actor_action_distributions, hx = create_training_batch(batch_of_trajectories)
        # If done, then set gamma to 0, else gamma
        discounts = (~dones).to(T.float32) * GAMMA

        optimizer.zero_grad()
        learner_action_distribution, values = learner_model(states, dones, hx,  actor=False)
        # This is used to create values_t_plus_1 during v-trace
        bootstrap_value = values[-1]

        v_trace_target, policy_grad_advantages = vtrace_target(actor_action_distributions,
                                                               learner_action_distribution,
                                                               actions_taken,
                                                               rewards,
                                                               values,
                                                               discounts,
                                                               bootstrap_value)

        # https://rlgraph.readthedocs.io/en/latest/reference/components/loss_functions_reference.html
        cross_entropy = F.cross_entropy(learner_action_distribution, actions_taken.to(T.long), reduction="none")
        # Baseline loss
        baseline_loss = BASELINE_COST * 0.5 * (v_trace_target - values).pow(2).sum()

        # Policy Gradient los
        policy_gradient_loss = (cross_entropy * policy_grad_advantages.detach()).sum()

        # Entropy loss
        policy = F.softmax(learner_action_distribution, 1)
        log_policy = F.log_softmax(learner_action_distribution, 1)
        entropy_per_timestep = (-policy * log_policy).sum(-1)
        entropy_loss = ENTROPY_COST * -entropy_per_timestep.sum()

        # Total loss
        loss = baseline_loss + policy_gradient_loss + entropy_loss
        loss.backward()

       # T.nn.utils.clip_grad_norm_(learner_model.parameters(), 40)

        optimizer.step()
        # Save learner model -> Push to parameter server
        parameter_server.push(learner_model.state_dict())
        latest_score = evaluator(learner_model.state_dict())
        scores.append(latest_score)
        if iteration > 100:
            running_average = np.sum(scores)/100
            print(loss.item(), running_average)


# Training batch is indexed by timestamp.
def create_training_batch(batch_of_trajectories):
    """
    states: Tensor of size trajectory_length * learner_batch_size *  * 1 * nS
    rewards: Tensor of size trajectory_length * learner_batch_size *
    actions: Tensor of size trajectory_length * learner_batch_size *
    actor_state_distributions: trajectory_length * nA * learner_batch_size
    """

    actions = []
    rewards = []
    dones = []
    states = []
    hx = []
    actor_action_distributions = []
    for trajectory in batch_of_trajectories:
        actions.append(trajectory.actions)
        rewards.append(trajectory.rewards)
        dones.append(trajectory.dones)
        states.append(trajectory.states)
        actor_action_distributions.append(trajectory.action_distributions)
        hx.append(trajectory.lstm_hx)
    states = T.stack(states).transpose(0, 1)
    actions = T.stack(actions).transpose(0, 1)
    rewards = T.stack(rewards).transpose(0, 1)
    dones = T.stack(dones).transpose(0, 1)
    # Why permute?
    actor_action_distributions = T.stack(actor_action_distributions).permute(1, 2, 0)
    hx = T.stack(hx).transpose(0, 1)
    return states, actions, rewards, dones, actor_action_distributions, hx
