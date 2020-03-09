import numpy as np


def select_action(state, already_taken_actions, transition_matrix):
    probabilities = transition_matrix[state]
    sorted_indices = np.argsort(probabilities)
    for i in sorted_indices:
        if i not in already_taken_actions:
            return i
    print("Sth wen wrong")