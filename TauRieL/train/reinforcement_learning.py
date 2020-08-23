import torch
import numpy as np
from train.distance import route_distance, tour_probability
from train.sampling import sample_episodes, random_episode, random_initialize_transition_matrix, \
    nearest_neighbour_initialize_transition_matrix, sample_episode_greedily
from model.actor_critic import Actor, Critic
import torch.optim as optim
from train.Data_Generator import TSPDataset
import pickle

from tqdm import tqdm

steps = 250
n = 20
dataset = TSPDataset(1, n)

points_list = dataset.data['Points_List']
solutions = dataset.data['Solutions']
first_cities = points_list[0]
first_solutions = solutions[0]


def reinforcement_learning(first_cities, baseline=0):
    phi, pi = random_episode(first_cities)
    l_pi = route_distance(phi)
    for t in range(steps):
        phi_samples = []
        for _ in range(250):
            phi_samples.append(sample_episode_greedily(first_cities, transition_matrix))
        lengths = [route_distance(first_cities[cities]) for cities in phi_samples]
        j = np.argmin(lengths)
        if lengths[j] < l_pi:
            l_pi = lengths[j]
            pi = phi_samples[j]

        reward = -lengths[j]
        avg_length = sum(lengths)/len(lengths)
        # update actor critic
        log_probs = torch.log(torch.from_numpy(np.array(tour_probability(phi_samples[j], transition_matrix))))

        advantage = torch.FloatTensor([-avg_length]) - torch.FloatTensor([baseline])
        # TODO check actor.forward(first_cities) in place log_probs
        # actor_loss = (log_probs * advantage).mean()
        actor_loss = (actor.forward(first_cities) * advantage).mean()
        # critic_loss = advantage.pow(2).mean()
        critic_loss = (critic.forward(first_cities) - avg_length*0.9).pow(2).mean()

        actor_optimizer.zero_grad()
        # actor_loss.requires_grad = True
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        # critic_loss.requires_grad = True
        critic_loss.backward()
        critic_optimizer.step()

        probabilities = actor(first_cities[phi_samples[j]])
        baseline = critic(first_cities)
        # probabilities = actor.forward(first_cities[phi_samples[j]])
        # baseline = critic.forward(first_cities[phi_samples[j]])
        if 0 == 0:
            for index in phi_samples[j]:
                j_index = np.argmax(transition_matrix[index])
                transition_matrix[index, j_index] = transition_matrix[index, j_index] + 1 * (
                            probabilities.detach().numpy()[index] - transition_matrix[index, j_index])  # sth else

        # if t == 0 or t == 50:
        #     for param in actor.parameters():
        #         print(param.data, param.names)

    return pi, transition_matrix, l_pi


data_iter = tqdm(dataset, unit='data')
c = np.array([[0, 0], [1, 0], [2, 0], [0, 1]])

actor = Actor(n*2, n)
critic = Critic(n*2)
actor_optimizer = optim.RMSprop(actor.parameters(), lr=3e-4)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=2e-4)
sum_pred = []
sum_exp = []
for i, data in enumerate(dataset):
    # data_iter.set_description(f'Cities {i}/{len(dataset)}')
    first_cities = data['Points']
    first_solutions = data['Solution']

    transition_matrix = nearest_neighbour_initialize_transition_matrix(first_cities)
    # print(f' initialized: \n  {transition_matrix}')
    s_t, t_m, r_d = reinforcement_learning(first_cities)
    sum_pred.append(r_d)
    sum_exp.append(route_distance(first_cities[first_solutions]))

    print(f'Shortest tour: {s_t}, length: {r_d}')
    print(f'Real shortest tour: {first_solutions}, {route_distance(first_cities[first_solutions])}')
    # print(f'Updated \n {transition_matrix}')
print(np.mean(sum_exp))
print(np.mean(sum_pred))
c = [[0.38082261, 0.62007715], [0.76648838, 0.42463141], [0.2885871, 0.18421117], [0.42288354, 0.57428997],
     [0.84263721, 0.02062535]]
s = [0, 2, 4, 1, 3]

#
# print(nearest_neighbour_initialize_transition_matrix(c))
