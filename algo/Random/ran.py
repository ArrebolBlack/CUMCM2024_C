import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import sys
import random

from env_new import CropPlantingEnv
sys.path.append('/home/shengjiaao/Newpython/C_2024')
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

def random_action(input_dim, num_years):
    """
    随机生成一个动作，动作空间为 [0, 1] 之间的连续值，
    """
    return np.random.uniform(low=0.0, high=1.0, size=(input_dim, num_years))

def run_random_episode(env, input_dim, num_years):
    """
    单个 episode 的执行函数，生成随机动作并返回奖励。
    """
    state = env.reset()
    action = random_action(input_dim, num_years)  # 随机选择动作
    next_state, reward, done, _ = env.step(action)
    return action, reward

def save_best_action(best_reward, best_action, save_dir, filename_prefix="best"):
    """
    保存当前最优的动作和奖励。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, f'{filename_prefix}_action.npy'), best_action)
    with open(os.path.join(save_dir, f'{filename_prefix}_reward.txt'), 'w') as f:
        f.write(str(best_reward))

    print(f"保存最优动作至 {save_dir}, 最优奖励: {best_reward}")

def parallel_random_exploration(env, input_dim, num_years, episodes_per_batch, num_batches, save_interval=10, save_dir="results"):
    """
    并行化的随机探索过程，使用多进程来处理多个 episodes。
    """
    rewards = []
    best_reward = -np.inf
    best_action = None
    avg_rewards = []

    for batch in tqdm(range(num_batches), desc="Training Batches"):
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(run_random_episode, [(env, input_dim, num_years) for _ in range(episodes_per_batch)])

        # 提取结果并更新最优奖励和动作
        batch_rewards = []
        for action, reward in results:
            rewards.append(reward)
            batch_rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_action = action
                # 发现新的最优动作，保存
                save_best_action(best_reward, best_action, save_dir=save_dir, filename_prefix=f'batch_{batch + 1}')

        avg_batch_reward = np.mean(batch_rewards)
        avg_rewards.append(avg_batch_reward)

        # 定期打印平均奖励
        if (batch + 1) % save_interval == 0:
            print(f"Batch {batch + 1}: Average Reward = {avg_batch_reward:.2f}, Best Reward = {best_reward:.2f}")

    return rewards, avg_rewards, best_reward, best_action

if __name__ == "__main__":
    # 初始化环境
    env = CropPlantingEnv()

    # 训练参数
    input_dim = 1082
    num_years = 7
    total_episodes = 100000
    import multiprocessing

    # 获取 CPU 核心数
    cpu_cores = multiprocessing.cpu_count()

    # 设置 batch_size 为核心数
    # batch_size = cpu_cores
    print(cpu_cores)
    batch_size = 100  # 每个批次的 episodes 数量
    num_batches = total_episodes // batch_size
    save_interval = 10  # 每隔多少个批次打印一次平均奖励

    # 并行执行随机探索
    save_dir = '/home/shengjiaao/Newpython/C_2024/algo/Random'
    rewards, avg_rewards, best_reward, best_action = parallel_random_exploration(env, input_dim, num_years, batch_size, num_batches, save_interval, save_dir)

    # 保存最终最优动作
    np.save(os.path.join(save_dir, 'final_best_action.npy'), best_action)
    print(f"最优动作已保存至 {save_dir}/final_best_action.npy, 最优奖励: {best_reward}")

    # 优化后的绘图，绘制总的 reward 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward", color='b', linewidth=1.5)
    plt.title('Random Exploration Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.show()
