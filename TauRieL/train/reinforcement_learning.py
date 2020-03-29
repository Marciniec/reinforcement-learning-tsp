import torch
import numpy as np
from train.distance import route_distance, tour_probability
from train.sampling import sample_episodes, random_episode, random_initialize_transition_matrix, \
    nearest_neighbour_initialize_transition_matrix
from model.actor_critic import Actor, Critic
import torch.optim as optim
from train.Data_Generator import TSPDataset
import pickle

from tqdm import tqdm

steps = 250
dataset = TSPDataset(1000, 10)

points_list = dataset.data['Points_List']
solutions = dataset.data['Solutions']
first_cities = points_list[0]
first_solutions = solutions[0]


def reinforcement_learning(first_cities, baseline=0):
    phi, pi = random_episode(first_cities)
    l_pi = route_distance(phi)
    for t in range(1000):
        phi_samples = []
        for i in range(len(first_cities)):
            phi_samples.append(sample_episodes(i, first_cities, transition_matrix))
        lengths = [route_distance(first_cities[cities]) for cities in phi_samples]
        j = np.argmin(lengths)
        if lengths[j] < l_pi:
            l_pi = lengths[j]
            pi = phi_samples[j]

        reward = lengths[j]

        # update actor critic
        log_probs = torch.log(torch.from_numpy(np.array(tour_probability(pi, transition_matrix))))

        advantage = torch.FloatTensor([reward]) - torch.FloatTensor([baseline])
        actor_loss = (log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()

        actor_optimizer.zero_grad()
        actor_loss.requires_grad = True
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.requires_grad = True
        critic_loss.backward()
        critic_optimizer.step()

        probabilities = actor.forward(first_cities[phi_samples[j]])
        baseline = critic.forward(first_cities[phi_samples[j]])
        if (t + 1) % 5 == 0:
            transition_matrix[:, j] = transition_matrix[:, j] + 0.01 * (
                    probabilities.detach().numpy() - transition_matrix[:, j])  # sth else

    return pi, transition_matrix, l_pi


data_iter = tqdm(dataset, unit='data')
c = np.array([[0, 0], [1, 0], [2, 0], [0, 1]])

actor = Actor(20, 10)
critic = Critic(20)
actor_optimizer = optim.RMSprop(actor.parameters(), lr=0.01)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=0.01)
sum_pred = []
sum_exp = []
for i, data in enumerate(dataset):
    # data_iter.set_description(f'Cities {i}/{len(dataset)}')
    first_cities = data['Points']
    first_solutions = data['Solution']

    transition_matrix = nearest_neighbour_initialize_transition_matrix(first_cities)
    # print(transition_matrix)
    s_t, t_m, r_d = reinforcement_learning(first_cities)
    sum_pred.append(r_d)
    sum_exp.append(route_distance(first_cities[first_solutions]))

    print(f'Shortest tour: {s_t}, length: {r_d}')
    print(f'Real shortest tour: {first_solutions}, {route_distance(first_cities[first_solutions])}')

print(np.mean(sum_exp))
print(np.mean(sum_pred))
c = [[0.38082261, 0.62007715], [0.76648838, 0.42463141], [0.2885871, 0.18421117], [0.42288354, 0.57428997],
     [0.84263721, 0.02062535]]
s = [0, 2, 4, 1, 3]

#
# print(nearest_neighbour_initialize_transition_matrix(c))
