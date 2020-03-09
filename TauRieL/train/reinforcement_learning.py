import torch
import numpy as np
from train.distance import route_distance, tour_probability
from train.sampling import sample_episodes, random_episode, initialize_transition_matrix
from model.actor_critic import ActorCritic
import torch.optim as optim
from train.Data_Generator import TSPDataset

dataset = TSPDataset(2, 5)
points_list = dataset.data['Points_List']
solutions = dataset.data['Solutions']
first_cities = points_list[0]
first_solutions = solutions[0]
baseline = 0

actor_critic = ActorCritic(1, 5, 4)
ac_optimizer = optim.Adam(actor_critic.parameters(), lr=0.01)
transition_matrix = initialize_transition_matrix(first_cities)

print(transition_matrix)


def reinforcement_learning(baseline=0):
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

        reward = -lengths[j]

        # update actor critic
        log_probs = torch.log(torch.from_numpy(np.array(tour_probability(pi, transition_matrix))))

        advantage = torch.FloatTensor([reward]) - torch.FloatTensor([baseline])
        actor_loss = (log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        ac_optimizer.zero_grad()
        ac_loss.requires_grad = True
        ac_loss.backward()
        ac_optimizer.step()

        baseline, probabilities = actor_critic.forward(np.array([j]))
        # if t % 7 == 0:
        transition_matrix[:, j] = probabilities.detach().numpy()

    return pi, transition_matrix, l_pi


s_t, t_m, r_d = reinforcement_learning()

print(f'Shortest tour: {s_t}, length: {r_d}  transition matrix: \n {t_m}')
print(f'Real shortest tour: {first_solutions}, {route_distance(first_cities[first_solutions])}')

c = [[0.38082261, 0.62007715], [0.76648838, 0.42463141], [0.2885871, 0.18421117], [0.42288354, 0.57428997],
     [0.84263721, 0.02062535]]
s = [0, 2, 4, 1, 3]
