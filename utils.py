import numpy as np


class ActionDiscretizer:
    def __init__(self, action_bounds, num_discrete_actions):
        """
        初始化动作离散化器

        参数:
        action_bounds: list of tuples, 每个动作的取值范围 [(min1, max1), (min2, max2), ...]
        num_discrete_actions: list, 每个动作维度的离散化数量 [n1, n2, ...]
        """
        self.action_bounds = action_bounds
        self.num_discrete_actions = num_discrete_actions
        self.discrete_actions = self._generate_discrete_actions()

    def _generate_discrete_actions(self):
        """
        生成离散化的动作空间

        返回:
        discrete_actions: np.array, 所有离散动作的组合
        """
        action_grids = []
        for (low, high), num in zip(self.action_bounds, self.num_discrete_actions):
            grid = np.linspace(low, high, num)  # 在范围内生成均匀分布的动作点
            action_grids.append(grid)

        # 生成离散化动作的笛卡尔积
        discrete_actions = np.array(np.meshgrid(*action_grids)).T.reshape(-1, len(self.action_bounds))
        return discrete_actions

    def get_discrete_action(self, continuous_action):
        """
        将连续动作映射到最接近的离散动作

        参数:
        continuous_action: np.array, 连续动作

        返回:
        discrete_action: np.array, 对应的离散动作
        """
        discrete_action = []
        for i, (low, high, num) in enumerate(zip(self.action_bounds, self.num_discrete_actions)):
            action_range = np.linspace(low, high, num)
            # 找到最接近的离散动作
            closest_idx = np.argmin(np.abs(action_range - continuous_action[i]))
            discrete_action.append(action_range[closest_idx])

        return np.array(discrete_action)

    def get_all_discrete_actions(self):
        """
        获取所有离散化的动作

        返回:
        discrete_actions: np.array, 所有可能的离散动作组合
        """
        return self.discrete_actions


# 示例使用
if __name__ == "__main__":
    # 定义动作的连续范围和每个维度的离散化数量
    action_bounds = [(-1, 1), (0, 10)]  # 动作范围: 第一个维度[-1, 1], 第二个维度[0, 10]
    num_discrete_actions = [5, 4]  # 离散化的数量: 第一个维度5个点, 第二个维度4个点

    discretizer = ActionDiscretizer(action_bounds, num_discrete_actions)

    # 打印所有可能的离散动作
    all_discrete_actions = discretizer.get_all_discrete_actions()
    print("所有离散化的动作组合:")
    print(all_discrete_actions)

    # 将连续动作映射到最近的离散动作
    continuous_action = np.array([0.3, 4.2])
    discrete_action = discretizer.get_discrete_action(continuous_action)
    print(f"连续动作 {continuous_action} 对应的离散动作 {discrete_action}")
