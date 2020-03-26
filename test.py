import gym
import torch

from algorithm.a2c import A2C
from algorithm.qlearn import DQN
from algorithm.mixture import Mixture

env_name = 'CartPole-v0'
algorithm_name = 'a2c'

env = gym.make(env_name)
seed = 123
env.seed(seed)
torch.manual_seed(seed)
log_interval = 10
lr = 2e-3

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print('Env Name: %s | Seed: %d | State_dim: %d | Action_dim: %d | Algo: %s '
      % (env_name, seed, state_dim, action_dim, algorithm_name))

model = DQN(state_dim, action_dim, lr=lr)


def main():
    running_reward = 0

    for i_episode in range(801):

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

        model.finish_episode()

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))


if __name__ == '__main__':
    main()