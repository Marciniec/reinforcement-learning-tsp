import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, number_of_inputs, number_of_cities):
        super(Actor, self).__init__()

        hidden_layer = 64
        hidden_layer_2 = 32
        self.actor = nn.Sequential(nn.Linear(number_of_inputs, hidden_layer),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer, hidden_layer_2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_2, hidden_layer_2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_2, hidden_layer_2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_2, hidden_layer_2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_2, hidden_layer_2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_2, number_of_cities),
                                   nn.Softmax()
                                   )

    def forward(self, permutations):
        x = permutations.flatten()
        update_vector_v = self.actor(x)

        return update_vector_v


class Critic(nn.Module):
    def __init__(self, number_of_inputs):
        super(Critic, self).__init__()

        hidden_layer = 64
        hidden_layer_2 = 32
        hidden_layer_3 = 16
        hidden_layer_4 = 16
        output_1 = 1

        self.critic = nn.Sequential(nn.Linear(number_of_inputs, hidden_layer),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer, hidden_layer_2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer_2, hidden_layer_2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer_2, hidden_layer_3),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer_3, hidden_layer_4),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer_4, output_1))

    def forward(self, permutations):
        x = permutations.flatten()
        baseline = self.critic(x)

        return baseline
