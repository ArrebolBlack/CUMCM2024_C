import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
import warnings
from eps_greedy import EpsilonGreedyAgent

# 忽略所有警告
warnings.filterwarnings("ignore")
sys.path.append('/home/shengjiaao/Newpython/C_2024')

def run_episode(env, agent):
    """
    单个 episode 的执行函数，用于在多进程中并行运行。
    """
    state = env.reset()
    action = agent.select_action()
    next_state, reward, done, _ = env.step(action)
    agent.update(action, reward)
    return action, reward

def parallel_training(env, agent, episodes_per_batch, num_batches):
    """
    并行化的训练过程，使用多进程来处理多个 episodes。
    """
    rewards = []
    best_reward = -np.inf
    best_action = None

    for batch in tqdm(range(num_batches), desc="Training Batches"):
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(run_episode, [(env, agent) for _ in range(episodes_per_batch)])

        for action, reward in results:
            rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_action = action

    return rewards, best_reward, best_action

if __name__ == "__main__":
    from env import CropPlantingEnv
  # 假设你的EpsilonGreedyAgent类定义在epsilon_greedy_agent.py中

    # 初始化环境和 Epsilon Greedy Agent
    env = CropPlantingEnv()
    agent = EpsilonGreedyAgent(num_plots=env.num_plots, num_crops=env.num_crops, num_years=env.num_years, epsilon=0.99)

    # 训练参数
    total_episodes = 10000000
    batch_size = 1000  # 每个批次的 episodes 数量
    num_batches = total_episodes // batch_size

    # 并行执行训练
    rewards, best_reward, best_action = parallel_training(env, agent, batch_size, num_batches)

    # 保存最优动作到文件
    np.save('best_action.npy', best_action)  # 将最优动作保存到文件
    print("最优动作已保存至 best_action.npy")

    # 绘制训练奖励曲线
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Epsilon-Greedy Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
