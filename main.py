import os
import time

import gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=False)
n_observation = env.observation_space.n
n_action = env.action_space.n


def value_iteration(env, gamma=1.0):
    ITERATION_COUNT = 1000
    EPS = 1e-18
    value = np.zeros(n_observation)
    for step in range(ITERATION_COUNT):
        next_value = np.copy(value)
        for now_state in range(n_observation):
            q_value = []
            for now_action in range(n_action):
                next_reward = []
                for state_probability, next_state, reward, terminated in env.P[now_state][now_action]:
                    next_reward.append(state_probability * (reward + gamma * next_value[next_state]))
                q_value.append(np.sum(next_reward))
                value[now_state] = max(q_value)
        if np.linalg.norm(next_value - value, ord=None) <= EPS:
            value = np.copy(next_value)
            break
        value = np.copy(next_value)
    return value


def extract_policy(env, gamma=1.0):


value_iteration(env)

env.close()
