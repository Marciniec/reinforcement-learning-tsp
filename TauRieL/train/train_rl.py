from model.actor_critic import Actor, Critic
from train.agent import Agent
from train.Data_Generator import TSPDataset
from torch.autograd import Variable
from train.distance import route_distance
import torch.optim as optim
from tqdm import tqdm

import functools
import torch
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    steps = 250
    batch_size = 250
    n_cities = 10
    n_examples = 1

    dataset = TSPDataset(n_examples, n_cities)
    points_list = dataset.data['Points_List']
    solutions = dataset.data['Solutions']
    first_cities = torch.from_numpy(points_list[0]).float()
    first_cities.to(device='cuda')
    first_solutions = solutions[0]

    actor = Actor(n_cities * 2, n_cities)
    actor_optimizer = optim.RMSprop(actor.parameters(), lr=3e-4)

    critic = Critic(n_cities * 2)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=2e-4)
    PATH = 'best_snapshot_devacc_0.946343668779931_devloss_2.502344846725464__iter_45450_model.pt'

    agent = Agent(PATH, batch_size)
    agent.init_transition_matrix_from_pointer_net(dataset)
    init_points = agent.sample_random_episode(first_cities)
    phi_samples_cities = []
    baseline = critic(init_points.permutation)

    data_iter = tqdm(range(steps), unit='steps')

    for i, _ in enumerate(data_iter):
        data_iter.set_description('steps %i/%i' % (i+1, steps))

        phi_samples_cities = agent.sample_episodes(first_cities)
        agent.calculate_shortest_tour_length(phi_samples_cities)

        log_probs = []
        lengths = []

        for sampled_points in phi_samples_cities:
            probability = functools.reduce(lambda x, y: x * y, sampled_points.probabilities)
            if probability > 1.0:
                raise Exception("Probability grater than one")
            log_probs.append(np.log(probability))
            lengths.append(sampled_points.length)

        _log_probs = Variable(torch.from_numpy(np.array(log_probs)), requires_grad=True)
        _lengths = Variable(torch.from_numpy(np.array(lengths)), requires_grad=True)

        advantage = _lengths - baseline

        actor_loss = (_log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        update_vector = actor(init_points.permutation)
        agent.update_transition_matrix(update_vector.detach().numpy(), init_points)
        baseline = critic(init_points.permutation)

    print(route_distance(first_cities[first_solutions]))
    print(agent.shortest_tour_length)
    print(agent.length_from_ptr_net)