import numpy as np


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
