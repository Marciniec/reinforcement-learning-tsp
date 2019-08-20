from data_generator import TSPDataset
from qlearning.q_learning import *

dataset = TSPDataset(200, 5)

q_learn(dataset,  500, 0.9, 0.01, 1, 0.001, 3)
