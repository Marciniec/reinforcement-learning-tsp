from train.action import select_action
import numpy as np


def sample_episodes(start_state, cities, transition_matrix):
    cities_number = 0
    current_state = start_state
    tour = [current_state]
    while cities_number < len(cities) - 1:
        current_state = select_action(current_state, tour, transition_matrix)
        tour.append(current_state)
        cities_number += 1
    return tour


def random_episode(cities):
    number_of_cities = len(cities)
    order = np.arange(number_of_cities)
    np.random.shuffle(order)
    return cities[order], order


def initialize_transition_matrix(cities):
    transition_matrix = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        transition_matrix[i] = np.insert(np.random.dirichlet(np.ones(len(cities) - 1), size=1), i, 0)
    return transition_matrix
