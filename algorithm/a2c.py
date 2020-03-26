import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from collections import namedtuple
from torch.distributions import Categorical
from algorithm.models import PVNet

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class A2C:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma

        self.action_dim = action_dim

        self.net = PVNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.saved_actions = []
        self.saved_rewards = []

    def select_action(self, state):
        s = torch.from_numpy(state).float()
        action_prob, state_value = self.net.forward(s)

        m = Categorical(action_prob)
        a = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(a), state_value))

        return a.item()

    def save_transition(self, state, action, reward, next_state, done):
        self.saved_rewards.append(reward)

    def finish_episode(self):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.saved_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        for (log_prob, state_value), R in zip(self.saved_actions, returns):
            advantage = R - state_value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(state_value, torch.tensor([R])))

        self.optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()

        self.optimizer.step()

        del self.saved_actions[:]
        del self.saved_rewards[:]
