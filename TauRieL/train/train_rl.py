from model.actor_critic import Actor, Critic
from train.agent import Agent
from train.Data_Generator import TSPDataset
from torch.autograd import Variable
from train.distance import route_distance
import torch.optim as optim
from tqdm import tqdm
from ptrnet.model_loader import ModelLoader
import functools
import torch
import numpy as np
from multiprocessing import freeze_support
import pickle
from torch.utils.data import DataLoader
from environment.environment import reward_func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    route_lengths_rl = []
    route_lengths_ptr = []
    route_lengths_3opt = []

    freeze_support()
    steps = 20
    batch_size = 64
    n_cities = 5
    n_examples = 1

    dataset = TSPDataset(n_examples, n_cities, solve=True)
    points_list = dataset.data['Points_List']
    solutions = dataset.data['Solutions']
    # pkl_file = open('dataset_15_50k.pickle', 'rb')
    # dataset = pickle.load(pkl_file)
    PATH = 'best_snapshot_devacc_0.946343668779931_devloss_2.502344846725464__iter_45450_model.pt'
    model_loader = ModelLoader(PATH)
    model_loader.load_model()
    # dataset.data['Solutions'] = np.array([0])
    data_points = model_loader.evaluate([np.roll(points_list[0], i, axis=0) for i in range(0, 5)])
    counter = 0
    pi_policy = []
    min_reward = np.inf
    for data_points_batch in data_points:
        for data_point in data_points_batch:
            first_cities, o, p = data_point
            first_cities = first_cities
            actor = Actor(n_cities, n_cities)
            actor.to(device)
            actor_optimizer = optim.RMSprop(actor.parameters(), lr=3e-4)

            critic = Critic(n_cities)
            critic.to(device)
            critic_optimizer = optim.RMSprop(critic.parameters(), lr=2e-4)

            agent = Agent(PATH, batch_size)
            agent.init_transition_matrix_from_pointer_net(data_point)
            init_points = agent.sample_random_episode(first_cities)
            init_points.calculate_length()
            min_reward = init_points.length
            pi_policy.append(init_points.indices)
            phi_samples_cities = []
            # baseline = critic(torch.from_numpy(init_points.permutation).float())
            for _ in tqdm(range(steps)):

                phi_samples_cities = agent.sample_episodes(first_cities)
                data_loader = DataLoader(phi_samples_cities,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=1)
                # agent.calculate_shortest_tour_length(phi_samples_cities)
                for batch_idx, batch in enumerate(data_loader):
                    indices, permutation = batch

                    indices = indices.to(device)
                    permutation = permutation.to(device)

                    dist, probs = actor(indices)
                    sample = dist.sample(torch.Size())
                    log_prob = dist.log_prob(sample)

                    reward = reward_func(permutation)
                    j_index = torch.argmin(reward)
                    if min_reward > reward[j_index]:
                        min_reward = reward[j_index]
                        pi_policy.append(indices[j_index])
                    critic_est = critic(indices).view(-1)

                    advantage = (reward - critic_est)
                    actor_loss = torch.mean(advantage.detach() * log_prob)
                    critic_loss = torch.mean(advantage ** 2)

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                with torch.no_grad():
                    _, update_vector = actor(torch.from_numpy(init_points.indices).float().to(device))
                    agent.update_transition_matrix(update_vector, init_points)

            counter += 1
            # route_lengths_3opt.append(route_distance(first_cities[first_solutions]))
            route_lengths_ptr.append(agent.length_from_ptr_net)
            route_lengths_rl.append(min_reward)

    # with open('route_lengths_3opt_50.pkl', 'wb') as pkl_file_route_lengths_3opt:
    #     pickle.dump(route_lengths_3opt, pkl_file_route_lengths_3opt)

    with open('route_lengths_ptr_50.pkl', 'wb') as pkl_file_route_lengths_ptr:
        pickle.dump(route_lengths_ptr, pkl_file_route_lengths_ptr)

    with open('route_lengths_rl_50.pkl', 'wb') as pkl_file_route_lengths_rl:
        pickle.dump(route_lengths_rl, pkl_file_route_lengths_rl)

    mean_3opt = np.array(route_lengths_3opt).mean()
    mean_ptr = np.array(route_lengths_ptr).mean()
    mean_rl = np.array(route_lengths_rl).mean()
    print(f'Mean rout length 3opt {mean_3opt}')
    print(f'Mean rout length ptr {mean_ptr}')
    print(f'Mean rout length rl {mean_rl}')
    print()
