import numpy as np
import torch


def reward_func(permutation):
    y = torch.cat((permutation, permutation[:, :1]), dim=1)
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    return tour_len.sum(1).detach()


class Environment:
    def route_distance(self, cities):
        distance = 0
        for i in range(1, len(cities)):
            a = cities[i - 1]
            b = cities[i]
            distance += np.linalg.norm(a - b)
        a = cities[len(cities) - 1]
        b = cities[0]
        distance += np.linalg.norm(a - b)
        return distance

    def step(self, cities_permutations):
        return -1 * self.route_distance(cities_permutations)
