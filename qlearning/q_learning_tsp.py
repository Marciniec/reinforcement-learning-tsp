from data_generator import TSPDataset
from qlearning.q_learning import *
from qlearning.big_map import Map

dataset = TSPDataset(200, 5)

map = Map(10)
distance_ = q_learn(dataset, 1000, 0.9, 0.01, 1, 0.001, 3, map)
print(len(distance_))
zeros_ = [d for d in distance_ if d == 0]
small_ = [d for d in distance_ if 0 < d < 0.01]

print(len(zeros_))

print(len(zeros_) / len(distance_))
print(len(small_) / len(distance_))
