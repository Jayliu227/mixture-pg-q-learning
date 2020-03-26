import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from algorithm.models import QFunc
from algorithm.RE import ReplayBuffer
from collections import namedtuple


class DQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma

        self.action_dim = action_dim

        self.net = QFunc(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        """ Hyperparameters """
        self.buffer_size = int(1e5)
        self.sample_size = 64

        self.eps_start = 1.0
        self.eps = self.eps_start
        self.eps_end = 0.01
        self.eps_decay = 0.995

        self.memory = ReplayBuffer(action_dim, self.buffer_size, self.sample_size)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            action_values = self.net.forward(state)
        self.net.train()

        if random.random() > self.eps:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def save_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def finish_episode(self):
        if len(self.memory) < self.sample_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        Q_targets_next = self.net.forward(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.net.forward(states).gather(1, actions)

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_targets, Q_expected)
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.eps_end, self.eps_decay * self.eps)