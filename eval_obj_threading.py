import numpy as np
import matplotlib.pyplot as plt
from Object_function import objective_function, planting_dispersion_decay
from env import CropPlantingEnv
from tqdm import tqdm
import concurrent.futures
import warnings

warnings.filterwarnings("ignore")  # 忽略所有警告

env = CropPlantingEnv()

# 动作空间的采样函数，采样一个 3 维矩阵
def sample_actions(action_space_shape, num_samples):
    # 在 3 维动作空间中随机采样，采样值范围在 [0, 1] 之间
    return np.random.uniform(low=0, high=1, size=(num_samples,) + action_space_shape)

# 单个目标函数的计算
def compute_reward(action):
    # return objective_function(action, env.yield_matrix, env.price_matrix, env.cost_matrix, env.sale_matrix, env.plot_areas)
    return planting_dispersion_decay(action)

# 分批处理的目标函数计算
def analyze_objective_distribution(action_space_shape, num_samples=1000, batch_size=100):
    rewards = []
    num_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    # 使用多线程而不是多进程
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 对于每个批次进行计算
        for batch_num in tqdm(range(num_batches + (1 if remainder > 0 else 0))):
            current_batch_size = batch_size if batch_num < num_batches else remainder

            if current_batch_size == 0:
                continue

            # 生成当前批次的动作
            actions = sample_actions(action_space_shape, current_batch_size)

            # 使用并行计算目标函数
            batch_rewards = list(executor.map(compute_reward, actions))

            rewards.extend(batch_rewards)

    rewards = np.array(rewards)

    # 分析目标函数的性质
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print(f"目标函数均值: {mean_reward}")
    print(f"目标函数标准差: {std_reward}")
    print(f"目标函数最小值: {min_reward}")
    print(f"目标函数最大值: {max_reward}")

    # 可视化目标函数的分布
    plt.hist(rewards, bins=30, alpha=0.7)
    plt.title("目标函数值分布")
    plt.xlabel("目标函数值")
    plt.ylabel("频率")
    plt.show()

# 示例使用，假设动作空间形状为 (82, 41, 7)，批次大小为 100
analyze_objective_distribution(action_space_shape=(82, 41, 7), num_samples=100, batch_size=10)
import gc
gc.collect()  # 手动释放内存
