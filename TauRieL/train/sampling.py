from train.action import select_action, select_action_greedily
import numpy as np


def sample_episodes(start_state, cities, transition_matrix):
    """Calculating tour for given start city based on transition matrix
       :returns array of integers representing indices of given cities"""
    cities_number = 0
    current_state = start_state
    tour = [current_state]
    while cities_number < len(cities) - 1:
        current_state = select_action(current_state, tour, transition_matrix)
        tour.append(current_state)
        cities_number += 1
    return tour


def sample_episode_greedily(cities, transition_matrix):
    cities_number = 0
    current_state = 0
    tour = [0]
    while cities_number < len(cities) - 1:
        current_state = select_action_greedily(current_state, tour, transition_matrix, cities)
        tour.append(current_state)
        cities_number += 1
    return tour


def random_episode(cities):
    """Returning permutation of city as list of integers representing indices of given cities"""
    number_of_cities = len(cities)
    order = np.arange(number_of_cities)
    np.random.shuffle(order)
    return cities[order], order


def random_initialize_transition_matrix(cities):
    """Returns transition matrix with random values where probability of cities sums to 1"""

    transition_matrix = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        transition_matrix[i] = np.insert(np.random.dirichlet(np.ones(len(cities) - 1), size=1), i, 0)
    return transition_matrix


def nearest_neighbour_initialize_transition_matrix(cities):
    """Returns transition matrix where closer city has higher probability"""

    transition_matrix = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        length_reciprocal = calculate_sum_reciprocal(i, cities)
        for j in range(len(cities)):
            if i == j:
                transition_matrix[i, j] = 0
            else:
                transition_matrix[i, j] = (1 / np.linalg.norm(cities[i] - cities[j])) / length_reciprocal
    return transition_matrix


def calculate_sum_reciprocal(index, cities):
    length = 0
    base_coordinates = cities[index]
    for i in range(len(cities)):
        if i == index:
            continue
        length += 1 / np.linalg.norm(base_coordinates - cities[i])
    return length
