import torch.nn as nn
import torch


class Policy(nn.Module):
    def __init__(self, number_of_states, number_of_actions):
        super(Policy, self).__init__()
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

        num_hidden = 128

        self.l1 = nn.Linear(number_of_states, num_hidden, bias=False)
        self.l2 = nn.Linear(num_hidden, number_of_actions, bias=False)

        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def reset(self):
        # Episode policy and reward history
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.5),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)
