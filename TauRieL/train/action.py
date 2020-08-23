import numpy as np


def select_action(state, already_taken_actions, transition_matrix):
    probabilities = transition_matrix[state]
    sorted_indices = np.argsort(probabilities)
    for i in sorted_indices:
        if i not in already_taken_actions:
            return i
    print("Sth wen wrong")


def select_action_greedily(state, already_taken_actions, transition_matrix, cities):
    probabilities = normalize_probabilities(transition_matrix[state])
    choice = _select_action_greedily(probabilities)
    for i in choice:
        if i not in already_taken_actions:
            return i
    print("Sth went wrong")


def select_action_greedily_probabilities(state, already_taken_actions, transition_matrix, cities):
    probabilities = transition_matrix[state]
    choice = _select_action_greedily(probabilities)
    for i in choice:
        if i not in already_taken_actions:
            return i, probabilities[i]
    print("Sth went wrong")


def _select_action_greedily(probabilities):
    choice = []
    tmp_probabilities = probabilities
    all = list(range(1, len(probabilities)))

    if sum(probabilities) == 0:
        return all
    for i in range(len(probabilities) - 1):
        tmp_probabilities = [x / sum(tmp_probabilities) for x in tmp_probabilities]
        random = np.random.rand()
        cumulative_sum = 0
        for n, p in enumerate(tmp_probabilities):
            cumulative_sum += p
            if cumulative_sum >= random:
                choice.append(n)
                tmp_probabilities[n] = 0
                break
    tmp_choice = set(choice)
    diff = set(all).difference(tmp_choice)
    return choice + list(diff)


def normalize_probabilities(probabilites):
    probabilities_sum = sum(probabilites)
    return probabilites / probabilities_sum
