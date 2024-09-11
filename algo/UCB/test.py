from ucb import UCBAgent
from env import CropPlantingEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Pool, cpu_count

import sys

sys.path.append('/home/shengjiaao/Newpython/C_2024')
import warnings

warnings.filterwarnings("ignore")  # 忽略所有警告


def run_episode(env, agent):
    """
    单个 episode 的执行函数，用于在多进程中并行运行。
    """
    state = env.reset()
    action = agent.select_action()
    next_state, reward, done, _ = env.step(action)
    agent.update(action, reward)
    return action, reward


def parallel_training(env, agent, episodes_per_batch, num_batches, processes):
    """
    并行化的训练过程，使用多进程来处理多个 episodes。
    增加 processes 参数控制 CPU 利用率。
    """
    rewards = []
    best_reward = -np.inf
    best_action = None
    best_rewards_per_batch = []

    for batch in tqdm(range(num_batches), desc="Training Batches"):
        with Pool(processes=processes) as pool:
            results = pool.starmap(run_episode, [(env, agent) for _ in range(episodes_per_batch)])

        batch_rewards = []
        for action, reward in results:
            batch_rewards.append(reward)
            rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        best_rewards_per_batch.append(best_reward)

        # 每个批次后保存当前最优模型
        np.save('best_action.npy', best_action)

        # 动态更新绘图
        plt.clf()
        plt.plot(rewards, label='Rewards')
        plt.plot([i * episodes_per_batch for i in range(len(best_rewards_per_batch))], best_rewards_per_batch,
                 label='Best Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('UCB Training Rewards')
        plt.legend()
        plt.pause(0.01)

    plt.show()
    return rewards, best_reward, best_action


if __name__ == "__main__":
    # 初始化环境和 UCB Agent
    env = CropPlantingEnv()
    agent = UCBAgent(num_plots=env.num_plots, num_crops=env.num_crops, num_years=env.num_years, c=100.0)

    # 训练参数
    total_episodes = 400000
    batch_size = 1000  # 每个批次的 episodes 数量
    num_batches = total_episodes // batch_size
    # processes = max(1, cpu_count() // 2)  # 使用一半的 CPU 核心进行训练
    processes = max(1, cpu_count())  # 使用一半的 CPU 核心进行训练
    # 并行执行训练
    rewards, best_reward, best_action = parallel_training(env, agent, batch_size, num_batches, processes)

    # 最终保存最优动作到文件
    np.save('best_action.npy', best_action)
    print("最优动作已保存至 best_action.npy")
  # 打印最佳奖励
    print(f"Best reward achieved: {best_reward}")