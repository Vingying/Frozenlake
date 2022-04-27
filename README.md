本项目基于 FrozenLake-v1 环境完成

## 环境介绍

FrozenLake-v1 环境提供了一个 size*size 大小的网格图，其中 ```S``` 表示起点，```G``` 表示宝箱，```H``` 表示洞，```F``` 表示冰面。其中冰面可以行走，玩家如果掉进洞中或者到达 ```G``` 则算一次 episode 结束。

环境对于移动方向进行映射，具体如下：

```
LEFT=0
DOWN=1
RIGHT=2
UP=3
```

玩家的目标是到达 ```G``` 并且不能掉进洞中。然而，冰面可能会很滑，所以玩家不一定每次都能够到达自己想要的位置。例如，假设玩家想要往上走，那么声明：

- 玩家向上走的概率是 1/3;
- 玩家向左走的概率是 1/3;
- 玩家向右走的概率是 1/3。

有打滑的设定，就需要玩家选择适当的策略来到达 ```G```。

### 设定回报

环境中设定的 Reward 如下：

- 如果玩家下一步到达的位置不是 ```G```，那么 Reward=0；
- 如果玩家下一步到达的位置是 ```G```，那么 Reward=1.

## 问题求解

由于环境、动作等都是已知的，于是该问题是一类 MDP 问题，问题就可以采用价值迭代的做法来求解。该算法分为价值迭代和策略提取两步，首先求解得到 Bellman 最优状态价值表，再根据此表求解对应的策略。

### 价值迭代（Value-Iteration）

Bellman 最优状态价值函数如下：
$$
v_{*}(s) = \max\limits_{a}\left[ R_{s}^a + \gamma \sum\limits_{s'\in \mathcal{S}}P_{ss'}^{a}v_{*}(s') \right]
$$
该部分代码如下：

```python
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
```

### 策略提取

该部分代码如下：

```python
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
```

## 求解结果

一个 episode 如下：

```python
env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True)  # 获得环境
```

![image-20220427212456314](C:\Users\Jingwei Yi\AppData\Roaming\Typora\typora-user-images\image-20220427212456314.png)

其中 ```is_slippery``` 表示该地图是否有打滑现象。

价值迭代进行了 1373 次收敛，对应的最优价值表和策略表如下

```
Converged at iteration #1373

optimal value table:
[0.82352941 0.82352941 0.82352941 0.82352941]
[0.82352941 0.         0.52941176 0.        ]
[0.82352941 0.82352941 0.76470588 0.        ]
[0.         0.88235294 0.94117647 0.        ]

optimal policy:
[0. 3. 3. 3.]
[0. 0. 0. 0.]
[3. 1. 0. 0.]
[0. 2. 1. 0.]
```