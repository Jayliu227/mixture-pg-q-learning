import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from torch.distributions import Categorical
from algorithm.models import QFunc
from algorithm.models import PVNet
from algorithm.RE import ReplayBuffer
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Mixture:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma

        self.action_dim = action_dim

        """ Q learning part """
        self.q_net = QFunc(state_dim, action_dim)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr * 2)

        """ Hyperparameters """
        self.buffer_size = int(1e4)
        self.sample_size = 128

        self.eps_start = 1.0
        self.eps = self.eps_start
        self.eps_end = 0.01
        self.eps_decay = 0.995

        self.memory = ReplayBuffer(action_dim, self.buffer_size, self.sample_size)

        """ Policy gradient part """
        self.p_net = PVNet(state_dim, action_dim)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), lr=lr)

        self.saved_actions = []
        self.saved_rewards = []
        self.saved_importance_weight = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)

        """ q learning part """
        self.q_net.eval()
        with torch.no_grad():
            action_values_q = self.q_net.forward(state)
        self.q_net.train()

        """ Policy gradient part """
        action_prob_p, state_value_p = self.p_net.forward(state)

        action_prob_mix = (F.softmax(action_values_q, dim=1) + action_prob_p) / 2

        policy_dist = Categorical(action_prob_p)
        mixture_dist = Categorical(action_prob_mix)

        if random.random() > self.eps:
            a = mixture_dist.sample()
        else:
            a = torch.tensor([random.randint(0, self.action_dim - 1)])

        self.saved_importance_weight.append(torch.exp(policy_dist.log_prob(a) - mixture_dist.log_prob(a)).detach())
        self.saved_actions.append(SavedAction(policy_dist.log_prob(a), state_value_p))
        return a.item()

    def save_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.saved_rewards.append(reward)

    def finish_episode(self):
        self.update_policy()
        if len(self.memory) >= self.sample_size:
            self.update_q()

    def update_policy(self):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.saved_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        for (log_prob, state_value), R, ip in zip(self.saved_actions, returns, self.saved_importance_weight):
            advantage = R - state_value.item()
            policy_losses.append(-log_prob * advantage * ip)
            value_losses.append(F.smooth_l1_loss(state_value.squeeze(0), torch.tensor([R])))

        self.p_optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()

        self.p_optimizer.step()

        del self.saved_actions[:]
        del self.saved_rewards[:]
        del self.saved_importance_weight[:]

    def update_q(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        Q_targets_next = self.q_net.forward(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.q_net.forward(states).gather(1, actions)

        self.q_optimizer.zero_grad()
        loss = F.mse_loss(Q_targets, Q_expected)
        loss.backward()
        self.q_optimizer.step()

        self.eps = max(self.eps_end, self.eps_decay * self.eps)
