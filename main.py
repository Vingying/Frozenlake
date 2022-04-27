import gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True)  # 获得环境
n_observation = env.observation_space.n  # 状态空间的总状态数
n_action = env.action_space.n  # 动作空间的总动作数
size = 4  # 地图大小，size*size


# 价值迭代
def value_iteration(environment, gamma=1.0):
    iteration_count = 10000  # 迭代最大次数
    eps = 1e-20  # 判断收敛的范围
    value = np.zeros(n_observation)  # 价值表
    for step in range(iteration_count):
        next_value = np.copy(value)  # 转移价值表
        for now_state in range(n_observation):
            q_value = []  # Q值
            for now_action in range(n_action):
                next_reward = []  # 转移回报
                for state_probability, next_state, reward, terminated in environment.P[now_state][now_action]:
                    next_reward.append(state_probability * (reward + gamma * next_value[next_state]))
                q_value.append(np.sum(next_reward))
                value[now_state] = max(q_value)
        if np.linalg.norm(value - next_value, ord=None) <= eps:  # 范数小于等于eps时判断为收敛
            print('Converged at iteration #' + str(step + 1) + "\n")
            break
    return value


# 根据最优价值表获得对应的策略表
def get_policy(value, gamma=1.0):
    policy = np.zeros(n_observation)
    for now_state in range(n_observation):
        q_table = np.zeros(n_action)
        for now_action in range(n_action):
            for state_probability, next_state, reward, terminated in env.P[now_state][now_action]:
                q_table[now_action] += state_probability * (reward + gamma * value[next_state])
        policy[now_state] = np.argmax(q_table)
    return policy


optimal_value_table = value_iteration(env, gamma=1.0)  # 最优价值表
optimal_policy = get_policy(optimal_value_table, gamma=1.0)  # 根据价值表得到的策略表
print('optimal value table:')
for i in range(size):
    print(optimal_value_table[i * 4: i * 4 + 4], sep=' ')
print('\noptimal policy:')
for i in range(size):
    print(optimal_policy[i * 4: i * 4 + 4], sep=' ')
env.close()
