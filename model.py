"""
This file contains the PyTorch code for the Vanilla DDQN and duel DDQN models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, total_states, total_actions, seed, flag):
        super(DQN, self).__init__()
        dim = 128
        self.flag = "duel"
        self.total_actions = total_actions
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(total_states, dim)
        if self.flag == "vanilla":
            self.fc2 = nn.Linear(dim, dim)
            self.fc3 = nn.Linear(dim, dim)
            self.fc4 = nn.Linear(dim, total_actions)
        else:
            self.fc2 = nn.Linear(dim, dim//2)
            self.fc3_A = nn.Linear(dim//2, dim//2)
            self.fc3_V = nn.Linear(dim//2, dim//2)
            self.fc4_A = nn.Linear(dim//2, total_actions)
            self.fc4_V = nn.Linear(dim//2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.flag == "vanilla":
            x = F.relu(self.fc3(x))
            return self.fc4(x)
        else:
            x = x.view(x.size(0), -1)

            adv = F.relu(self.fc3_A(x))
            val = F.relu(self.fc3_V(x))

            adv = self.fc4_A(adv)
            val = self.fc4_V(val).expand(x.size(0), self.total_actions)

            x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.total_actions)

        return x
