from env import CropPlantingEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import sys
sys.path.append('/home/shengjiaao/Newpython/C_2024')
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

class RandomAgent:
    def __init__(self, num_plots, num_crops, num_years):
        self.num_plots = num_plots
        self.num_crops = num_crops
        self.num_years = num_years

    def select_action(self):
        # 在动作空间中随机选择一个动作
        return np.random.uniform(0, 1, (self.num_plots, self.num_crops, self.num_years))

def run_episode(env, agent):
    """
    单个 episode 的执行函数，用于在多进程中并行运行。
    """
    state = env.reset()
    action = agent.select_action()
    next_state, reward, done, _ = env.step(action)
    return action, reward

def parallel_training(env, agent, episodes_per_batch, num_batches):
    """
    并行化的训练过程，使用多进程来处理多个 episodes。
    """
    best_reward = -np.inf
    best_action = None

    for batch in tqdm(range(num_batches), desc="Training Batches"):
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(run_episode, [(env, agent) for _ in range(episodes_per_batch)])

        for action, reward in results:
            if reward > best_reward:
                best_reward = reward
                best_action = action

                # 实时打印当前的最高reward并保存动作
                print(f"当前最高的reward: {best_reward}")
                np.save('best_action.npy', best_action)  # 实时保存最优动作到文件

    return best_reward, best_action

if __name__ == "__main__":
    # 初始化环境和 Random Agent
    env = CropPlantingEnv()
    agent = RandomAgent(num_plots=env.num_plots, num_crops=env.num_crops, num_years=env.num_years)

    # 训练参数
    total_episodes = 20480000
    batch_size = 512  # 每个批次的 episodes 数量
    num_batches = total_episodes // batch_size

    # 并行执行训练
    best_reward, best_action = parallel_training(env, agent, batch_size, num_batches)

    # 最终输出最高的reward和对应的动作
    print(f"最终最高的reward: {best_reward}")
    print(f"对应的最优动作: {best_action}")
    print("最优动作已保存至 best_action.npy")
