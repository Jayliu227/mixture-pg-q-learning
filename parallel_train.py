import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import itertools
from scipy.interpolate import interp1d

from algorithm.a2c import A2C
from algorithm.mixture import Mixture
from algorithm.qlearn import DQN

log_interval = 10
env_name = 'CartPole-v0'
gamma = 0.99
lr = 5e-3


def worker(worker_id, algorithm_name, seed, return_dict):
    print('Worker %d (pid: %d) has started: algorithm_name <%s> seed <%d>.' % (
        worker_id, os.getpid(), algorithm_name, seed))
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if algorithm_name == 'a2c':
        model = A2C(state_dim, action_dim, lr=lr, gamma=gamma)
    elif algorithm_name == 'mix':
        model = Mixture(state_dim, action_dim, lr=lr, gamma=gamma)
    elif algorithm_name == 'dqn':
        model = DQN(state_dim, action_dim, lr=lr, gamma=gamma)
    else:
        raise NotImplementedError('Not such algorithm.')

    reward_records = []

    running_reward = 0

    for i_episode in range(501):

        state = env.reset()
        ep_reward = 0

        for t in range(1, 10000):

            action = model.select_action(state)

            next_state, reward, done, _ = env.step(action)

            model.save_transition(state, action, reward, next_state, done)

            state = next_state

            ep_reward += reward

            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        reward_records.append(running_reward)

        model.finish_episode()

        if i_episode % log_interval == 0:
            print('{:>10}(pid:{:>4}-worker_id:{:>2})|Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                algorithm_name, seed, worker_id, i_episode, ep_reward, running_reward))

    env.close()
    return_dict[worker_id] = reward_records
    print('Worker %d has ended.' % worker_id)


def plot(rewards, algo_names):
    x = np.array([i for i in range(len(rewards[0]))])
    colors = ['C%d' % i for i in range(len(algo_names))]

    num_of_exp = len(rewards) // len(algo_names)

    for i in range(len(algo_names)):
        reward = np.array(rewards[i * num_of_exp: (i + 1) * num_of_exp])

        mean = reward.mean(axis=0)
        # mean_itp = interp1d(x, mean, kind='quadratic', fill_value='extrapolate')
        median = np.median(reward, axis=0)
        std = reward.std(axis=0)

        # plt.plot(x, mean_itp(mean), lw=2)
        plt.plot(x, mean, '-', lw=2, color=colors[i])
        plt.plot(x, median, '--', lw=1, color=colors[i], alpha=0.8)
        # plt.fill_between(x, mean_itp(mean) - std, mean_itp(mean) + std, alpha=0.3)
        plt.fill_between(x, mean - std, mean + std, facecolor=colors[i], alpha=0.3)

    plt.title('Plot of Rewards Averaged over %d Trials' % num_of_exp)
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.legend([x + y for (x, y) in itertools.product(algo_names, ['(mean)', '(median)'])])
    plt.grid()

    plt.savefig('./plots/%s_[%s].png' % (env_name, ''.join(algo_names)), dpi=200)
    plt.show()


def main():
    manager = mp.Manager()
    return_dict = manager.dict()

    algo_names = ['a2c', 'mix']
    seeds = [23, 222, 1133, 444, 555]

    num_algorithms = len(algo_names)
    names = []
    for i in range(num_algorithms):
        names += [algo_names[i]] * len(seeds)
    seeds = seeds * num_algorithms

    processes = []

    for i in range(len(seeds)):
        p = mp.Process(target=worker, args=(i, names[i], seeds[i], return_dict))
        processes.append(p)
        p.start()

    for i in processes:
        i.join()

    result = [return_dict[key] for key in sorted(return_dict.keys(), reverse=False)]
    plot(result, algo_names)


if __name__ == '__main__':
    main()