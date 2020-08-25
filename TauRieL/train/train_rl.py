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

if __name__ == '__main__':
    route_lengths_rl = []
    route_lengths_ptr = []
    route_lengths_3opt = []

    freeze_support()
    steps = 250
    batch_size = 4
    n_cities = 15
    n_examples = 300_000

    # dataset = TSPDataset(n_examples, n_cities)
    # points_list = dataset.data['Points_List']
    # solutions = dataset.data['Solutions']
    pkl_file = open('dataset_15_50k.pickle', 'rb')
    dataset = pickle.load(pkl_file)
    PATH = 'best_snapshot_devacc_0.946343668779931_devloss_2.502344846725464__iter_45450_model.pt'
    model_loader = ModelLoader(PATH)
    model_loader.load_model()
    data_points = model_loader.evaluate(dataset)
    counter = 0
    for data_points_batch in data_points:
        if counter > 100:
            break
        for data_point in data_points_batch:
            first_cities, first_solutions, o, p = data_point
            first_solutions = first_solutions.cpu().detach().numpy()
            first_cities = first_cities.cpu().detach().numpy()
            actor = Actor(n_cities * 2, n_cities)
            actor_optimizer = optim.RMSprop(actor.parameters(), lr=3e-4)

            critic = Critic(n_cities * 2)
            critic_optimizer = optim.RMSprop(critic.parameters(), lr=2e-4)

            agent = Agent(PATH, batch_size)
            agent.init_transition_matrix_from_pointer_net(data_point)
            init_points = agent.sample_random_episode(first_cities)
            phi_samples_cities = []
            baseline = critic(torch.from_numpy(init_points.permutation).float())

            data_iter = tqdm(range(steps), unit='steps')

            for i, _ in enumerate(data_iter):
                data_iter.set_description('steps %i/%i' % (i + 1, steps))

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

                update_vector = actor(torch.from_numpy(init_points.permutation).float())
                agent.update_transition_matrix(update_vector.detach().numpy(), init_points)
                baseline = critic(torch.from_numpy(init_points.permutation).float())
            counter += 1
            route_lengths_3opt.append(route_distance(first_cities[first_solutions]))
            route_lengths_ptr.append(agent.length_from_ptr_net)
            route_lengths_rl.append(agent.shortest_tour_length)

    with open('route_lengths_3opt.pkl', 'wb') as pkl_file_route_lengths_3opt:
        pickle.dump(route_lengths_3opt, pkl_file_route_lengths_3opt)

    with open('route_lengths_ptr.pkl', 'wb') as pkl_file_route_lengths_ptr:
        pickle.dump(route_lengths_ptr, pkl_file_route_lengths_ptr)

    with open('route_lengths_rl.pkl', 'wb') as pkl_file_route_lengths_rl:
        pickle.dump(route_lengths_rl, pkl_file_route_lengths_rl)

    mean_3opt = np.array(route_lengths_3opt).mean()
    mean_ptr = np.array(route_lengths_ptr).mean()
    mean_rl = np.array(route_lengths_rl).mean()
    print(f'Mean rout length 3opt {mean_3opt}')
    print(f'Mean rout length ptr {mean_ptr}')
    print(f'Mean rout length rl {mean_rl}')
