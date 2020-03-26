import torch.nn as nn
import torch.nn.functional as F


class PVNet(nn.Module):
    """ Policy network and value network combined """

    def __init__(self, state_dim, action_dim):
        super(PVNet, self).__init__()

        self.affine1 = nn.Linear(state_dim, 32)
        self.affine2 = nn.Linear(32, 64)

        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_probs, state_values

    def calc_v(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        state_values = self.value_head(x)

        return state_values


class QFunc(nn.Module):
    def __init__(self, state_dim, action_dim):
        """ single action for each state """
        super(QFunc, self).__init__()
        self.affine1 = nn.Linear(state_dim, 32)
        self.affine2 = nn.Linear(32, 64)
        self.affine3 = nn.Linear(64, action_dim)

    def forward(self, x):
        """ get Q(s, a) for all a """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.affine3(x)
        return x
