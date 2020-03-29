import numpy as np


def route_distance(cities):
    distance = 0
    for i in range(1, len(cities)):
        a = cities[i - 1]
        b = cities[i]
        distance += np.linalg.norm(a - b)
    a = cities[len(cities) - 1]
    b = cities[0]
    distance += np.linalg.norm(a - b)
    return distance


def tour_probability(cities, transition_matrix):
    tour_probability_ = 1
    for i in range(1, len(cities)):
        n_city, n_prev_city = cities[i], cities[i - 1]
        tour_probability_ *= transition_matrix[n_prev_city, n_city]
    return tour_probability_
