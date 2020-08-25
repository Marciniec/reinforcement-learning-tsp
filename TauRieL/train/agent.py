from environment.points import Points
import numpy as np
from train.sampling import random_episode
from train.distance import route_distance
from train.sampling import sample_episode_greedily_with_probabilities
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from train.PointerNet import PointerNet


class Agent:
    def __init__(self, model_path, batch_size):
        self.transition_matrix = None
        self.epsilon = 0.01
        self.shortest_tour_length = None
        self.policy = None
        self.model_path = model_path
        self.model_batch_size = 256
        self.batch_size = batch_size
        self.tour_from_ptr_net = None
        self.length_from_ptr_net = None

    def sample_random_episode(self, input_city_permutation):
        permutation, order = random_episode(input_city_permutation)
        self.shortest_tour_length = route_distance(permutation)
        return Points(order, permutation, input_city_permutation)

    def sample_episodes(self, input_city_permutation):
        phi_samples_cities = []
        for _ in range(self.batch_size):
            phi_samples_cities_indices, phi_samples_cities_probabilities = sample_episode_greedily_with_probabilities(
                input_city_permutation,
                self.transition_matrix)
            sampled_points = Points(phi_samples_cities_indices, input_city_permutation[phi_samples_cities_indices],
                                    input_city_permutation, phi_samples_cities_probabilities)
            sampled_points.calculate_length()
            phi_samples_cities.append(sampled_points)
        return phi_samples_cities

    def calculate_shortest_tour_length(self, points: [Points]):
        points_ = sorted(points, key=lambda x: x.length)[0]
        if self.shortest_tour_length is None or self.shortest_tour_length > points_.length:
            self.shortest_tour_length = points_.length

        return points_

    def init_transition_matrix(self, input_city_permutation):
        transition_matrix = np.zeros((len(input_city_permutation), len(input_city_permutation)))
        for i in range(len(input_city_permutation)):
            length_reciprocal = calculate_sum_reciprocal(i, input_city_permutation)
            for j in range(len(input_city_permutation)):
                if i == j:
                    transition_matrix[i, j] = 0
                else:
                    transition_matrix[i, j] = (1 / np.linalg.norm(
                        input_city_permutation[i] - input_city_permutation[j])) / length_reciprocal
        self.transition_matrix = transition_matrix

    def init_transition_matrix_from_pointer_net(self, data_points):
        input_city_permutation, ptrnet_solution, o, p = data_points
        transition_matrix = np.zeros((len(input_city_permutation), len(input_city_permutation)))
        probs = o.contiguous().view(-1, o.size()[-1]).cpu().detach().numpy()
        next_index = 0
        for i in range(1, len(probs)):
            transition_matrix[next_index] = probs[i]
            next_index = np.argmax(probs[i])
        self.transition_matrix = transition_matrix
        self.tour_from_ptr_net = p.contiguous().view(-1, p.size()[-1]).cpu().detach().numpy()[0]
        self.length_from_ptr_net = route_distance(input_city_permutation[self.tour_from_ptr_net].cpu().detach().numpy())
        return

    def update_transition_matrix(self, update_vector: np.ndarray, points: Points):
        for index in points.indices:
            j_index = np.argmax(self.transition_matrix[index])
            self.transition_matrix[index, j_index] = self.transition_matrix[index, j_index] + self.epsilon * (
                    update_vector[index] - self.transition_matrix[index, j_index])


def calculate_sum_reciprocal(index, cities):
    length = 0
    base_coordinates = cities[index]
    for i in range(len(cities)):
        if i == index:
            continue
        length += 1 / np.linalg.norm(base_coordinates - cities[i])
    return length
