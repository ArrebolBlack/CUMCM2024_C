
class SimpleUCBAgent:
    def __init__(self, input_dim, num_years, c=1.0, resolution=0.1):
        self.input_dim = input_dim  # 动作空间中的输入维度，如 859
        self.num_years = num_years  # 时间维度，如 7 年
        self.c = c  # 控制探索与利用的平衡
        self.resolution = resolution  # 动作离散化的分辨率，如 0.1
        self.num_discrete_actions = int(1.0 / self.resolution) + 1  # 离散动作的数量为 11（0.0, 0.1, ..., 1.0）
        self.action_space = np.linspace(0, 1, self.num_discrete_actions)  # 离散化后的动作空间

        # 记录每个动作的选择次数和奖励
        self.counts = np.zeros((self.input_dim, self.num_discrete_actions, self.num_years))
        self.values = np.zeros((self.input_dim, self.num_discrete_actions, self.num_years))

    def select_action(self):
        """使用 UCB 公式选择动作"""
        total_counts = np.sum(self.counts) + 1  # 总的选择次数，防止除以0
        ucb_values = self.values + self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-5))

        # 为每一年选择 859 个动作
        action = np.zeros((self.input_dim, self.num_years))

        for year in range(self.num_years):
            for i in range(self.input_dim):
                # 为每一年选择 859 个离散动作，每个动作的 UCB 值最大化
                best_action_idx = np.argmax(ucb_values[i, :, year])
                action[i, year] = self.action_space[best_action_idx]  # 将离散动作的索引映射为动作值

        return action

    def update(self, action, reward):
        """更新每个动作的选择次数和奖励值"""
        for i in range(self.input_dim):
            for year in range(self.num_years):
                # 找到实际执行的动作在离散动作中的索引
                closest_action_idx = np.argmin(np.abs(self.action_space - action[i, year]))
                self.counts[i, closest_action_idx, year] += 1  # 更新该动作的选择次数
                # 使用增量平均公式更新奖励值
                self.values[i, closest_action_idx, year] += (reward - self.values[i, closest_action_idx, year]) / \
                                                            self.counts[i, closest_action_idx, year]

    def reset(self):
        """重置 agent 的状态"""
        self.counts.fill(0)
        self.values.fill(0)


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ContinuousUCBAgent:
    def __init__(self, input_dim, num_years, c=1.0):
        self.input_dim = input_dim  # 859
        self.num_years = num_years  # 7
        self.c = c  # 控制探索与利用的平衡
        self.action_min = 0.0  # 动作最小值
        self.action_max = 1.0  # 动作最大值

        # 为每个输入维度和年份创建一个高斯过程回归模型
        self.models = [[self.create_gp_model() for _ in range(num_years)] for _ in range(input_dim)]
        self.actions_taken = []  # 存储已采取的动作
        self.rewards = []  # 存储对应的奖励

    def create_gp_model(self):
        """创建一个高斯过程回归模型"""
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

    def select_action(self):
        """通过 UCB 公式选择动作"""
        action = np.zeros((self.input_dim, self.num_years))

        for i in range(self.input_dim):
            for year in range(self.num_years):
                # 从高斯过程回归模型中预测均值和标准差
                if len(self.actions_taken) > 0:
                    X = np.array(self.actions_taken)[:, i, year].reshape(-1, 1)
                    y = np.array(self.rewards).reshape(-1, 1)

                    # 更新高斯过程模型
                    self.models[i][year].fit(X, y)

                # 为当前维度和年份生成一系列候选动作
                candidate_actions = np.linspace(self.action_min, self.action_max, 100).reshape(-1, 1)

                # 预测候选动作的均值和标准差
                means, stds = self.models[i][year].predict(candidate_actions, return_std=True)

                # 计算 UCB 值
                ucb_values = means + self.c * stds

                # 选择 UCB 值最大的动作
                best_action_idx = np.argmax(ucb_values)
                best_action = candidate_actions[best_action_idx]

                # 将最佳动作存入 action 矩阵
                action[i, year] = best_action

        return action

    def update(self, action, reward):
        """更新模型的训练数据"""
        self.actions_taken.append(action)
        self.rewards.append(reward)

    def reset(self):
        """重置 agent 的状态"""
        self.actions_taken = []
        self.rewards = []
        # 重新创建高斯过程模型
        self.models = [[self.create_gp_model() for _ in range(self.num_years)] for _ in range(self.input_dim)]
