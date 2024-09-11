from algo.UCB.UCB_new_env.ucb import SimpleUCBAgent
from ucb import ContinuousUCBAgent, SimpleUCBAgent
from env_new import CropPlantingEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

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


def parallel_training(env, agent, episodes_per_batch, num_batches, save_interval=10, save_dir="results"):
    """
    并行化的训练过程，使用多进程来处理多个 episodes。
    """
    rewards = []
    best_reward = -np.inf
    best_action = None
    avg_rewards = []

    for batch in tqdm(range(num_batches), desc="Training Batches"):
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(run_episode, [(env, agent) for _ in range(episodes_per_batch)])

        # 提取结果并更新最优奖励和动作
        batch_rewards = []
        for action, reward in results:
            rewards.append(reward)
            batch_rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_action = action
                # # 发现新的最优动作，保存
                # save_best_action(best_reward, best_action, save_dir=save_dir, filename_prefix=f'batch_{batch + 1}')

        avg_batch_reward = np.mean(batch_rewards)
        avg_rewards.append(avg_batch_reward)

        # 定期打印平均奖励
        if (batch + 1) % save_interval == 0:
            print(f"Batch {batch + 1}: Average Reward = {avg_batch_reward:.2f}, Best Reward = {best_reward:.2f}")

    return rewards, avg_rewards, best_reward, best_action


if __name__ == "__main__":
    # 初始化环境和 UCB Agent
    env = CropPlantingEnv()
    agent = SimpleUCBAgent(input_dim=1082, num_years=7, c=100.0, resolution=0.1)

    # 训练参数
    total_episodes = 1000
    batch_size = 100  # 每个批次的 episodes 数量
    num_batches = total_episodes // batch_size
    save_interval = 10  # 每隔多少个批次打印一次平均奖励

    # 并行执行训练
    save_dir = '/home/shengjiaao/Newpython/C_2024/algo/UCB/UCB_new_env/result'
    rewards, avg_rewards, best_reward, best_action = parallel_training(env, agent, batch_size, num_batches,
                                                                       save_interval, save_dir)

    # 保存最终最优动作
    np.save(os.path.join(save_dir, 'final_best_action.npy'), best_action)
    print(f"最优动作已保存至 {save_dir}/final_best_action.npy, 最优奖励: {best_reward}")

    # 优化后的绘图，绘制总的 reward 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward", color='b', linewidth=1.5)
    plt.title('UCB Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.show()

    # # 绘制训练奖励曲线
    # plt.figure(figsize=(12, 6))
    #
    # # 绘制单个 episode 的奖励曲线
    # plt.subplot(1, 2, 1)
    # plt.plot(rewards)
    # plt.title('UCB Training Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    #
    # # 绘制每个批次的平均奖励曲线
    # plt.subplot(1, 2, 2)
    # plt.plot(avg_rewards)
    # plt.title('Average Reward per Batch')
    # plt.xlabel('Batch')
    # plt.ylabel('Average Reward')
    #
    # plt.tight_layout()
    # plt.show()
