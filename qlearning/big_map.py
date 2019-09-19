import numpy as np


class Map:
    def __init__(self, number_of_buckets):
        self.number_of_buckets = number_of_buckets
        self.map = self._init_map()
        self.world = self._init_empty_world()

    def _init_map(self):
        map = []
        for i in range(0, self.number_of_buckets):
            sub_map = []
            for j in range(0, self.number_of_buckets):
                sub_map.append(i * self.number_of_buckets + j)
            map.append(sub_map)
        return np.array(map)

    def _init_empty_world(self):
        world = []
        for i in range(0, self.number_of_buckets ** 2):
            sub_world = []
            for j in range(0, self.number_of_buckets ** 2):
                sub_world.append(0)
            world.append(sub_world)
        return np.array(world,  dtype=np.float64)

    def put(self, x: float, y: float, value: float):
        self.map[int(x * self.number_of_buckets)][int(y * self.number_of_buckets)] = value

    def get_action(self, x: float, y: float):
        return self.map[int(x * self.number_of_buckets)][int(y * self.number_of_buckets)]

    def put_rewards(self, x_1: float, y_1: float, x_2: float, y_2: float, rewards: float):
        action_1 = self.get_action(x_1, y_1)
        action_2 = self.get_action(x_2, y_2)
        self.world[action_1][action_2] = rewards

    def get_rewards(self, x_1, y_1, x_2, y_2):
        action_1 = self.get_action(x_1, y_1)
        action_2 = self.get_action(x_2, y_2)
        if action_1 == action_2:
            return 0.
        return self.world[action_1][action_2]
