import numpy as np


class EpsilonGreedyAgent:
    def __init__(self, num_plots, num_crops, num_years, epsilon=0.1):
        self.num_plots = num_plots
        self.num_crops = num_crops
        self.num_years = num_years
        self.epsilon = epsilon  # 控制探索与利用的平衡
        self.counts = np.zeros((num_plots, num_crops, num_years))  # 记录每个动作的选择次数
        self.values = np.zeros((num_plots, num_crops, num_years))  # 记录每个动作的平均奖励

    def select_action(self):
        # 随机探索：以概率 epsilon 随机选择动作
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_crops, size=(self.num_plots, self.num_years))
        else:
            # 利用现有信息选择最优动作（利用率高的选择）
            action = np.argmax(self.values, axis=1)

        # 转换为one-hot编码，保持与self.num_plots, self.num_crops, self.num_years一致的形状
        action_one_hot = np.zeros((self.num_plots, self.num_crops, self.num_years))
        for plot in range(self.num_plots):
            for year in range(self.num_years):
                action_one_hot[plot, action[plot, year], year] = 1

        return action_one_hot

    def update(self, action, reward):
        # 取 action 中每个 plot-crop-year 的值，并更新相应的奖励值
        for plot in range(self.num_plots):
            for crop in range(self.num_crops):
                for year in range(self.num_years):
                    if action[plot, crop, year] == 1:  # 只有选择了该作物，才更新
                        self.counts[plot, crop, year] += 1  # 更新选择次数
                        # 更新平均奖励
                        self.values[plot, crop, year] += (reward - self.values[plot, crop, year]) / self.counts[
                            plot, crop, year]

    def set_epsilon(self, new_epsilon):
        """更新 epsilon 值，动态调整探索与利用的平衡"""
        self.epsilon = new_epsilon
