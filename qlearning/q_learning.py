from math import sqrt
import numpy as np


def reward(dataset, state_index, other_index):
    return sqrt((dataset[state_index][0] - dataset[other_index][0]) ** 2 + (
            dataset[state_index][1] - dataset[other_index][1]) ** 2)


def create_reward_table(dataset, goal_reward):
    R = np.array([[0.] * (len(dataset) + 1)] * (len(dataset) + 1))
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset)):
            R[i, j] = -reward(dataset, i, j)
    for i in range(1, len(dataset)):
        R[i, len(dataset)] = goal_reward - reward(dataset, 0, i)
    return R


def epsilon_greedy(q, state, actions, epsilon):
    if np.random.rand() > epsilon:
        max_q = np.max(q[state, actions])
        greedy_actions = [i for i, j in enumerate(q[state, :]) if j == max_q and i in actions]
        action = np.random.choice(greedy_actions)
    else:
        action = np.random.choice(actions)
    return action


def q_learn(dataset, epochs, alpha, gamma, epsilon, epsilon_decay, goal_reward):
    points, solutions = dataset.data['Points_List'], dataset.data['Solutions']
    Q = np.array([[0.] * (len(points[0]) + 1)] * (len(points[0]) + 1))

    for index, points_set in enumerate(points):
        total_cost, transitions, Q = _q_learn(points_set, Q, epochs, alpha, gamma, epsilon, epsilon_decay, goal_reward)
        print(transitions)
        print(solutions[index])
    print(Q)


def _q_learn(points, Q, epochs, alpha, gamma, epsilon, epsilon_decay, goal_reward):
    start_state = 0
    end_state = len(points)
    transitions = {}
    total_cost = []
    R = create_reward_table(points, goal_reward)

    for i in range(0, epochs):
        possible_actions = np.arange(1, len(points))
        state = start_state
        cost = 0
        transitions[i] = [state]
        goal = False

        while not goal:
            action = epsilon_greedy(Q, state, possible_actions, epsilon)
            next_state = action
            possible_actions = possible_actions[possible_actions != action]

            if len(possible_actions) == 0:
                possible_actions = np.array([end_state])

            Q[state, action] = Q[state, action] + alpha * (
                    (R[state, action] + gamma * max(Q[next_state, possible_actions])) - Q[state, action])
            cost += R[state, action]
            state = next_state
            if state == end_state:
                break
            else:
                transitions[i].append(action)
            epsilon -= epsilon_decay * epsilon

        total_cost.append(cost)
    return total_cost, transitions[499], Q
