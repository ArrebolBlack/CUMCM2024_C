from ucb import UCBAgent
from env import CropPlantingEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 主训练过程
if __name__ == "__main__":
    # 初始化环境和 UCB Agent
    env = CropPlantingEnv()
    agent = UCBAgent(num_plots=env.num_plots, num_crops=env.num_crops, num_years=env.num_years, c=100.0)

    # 训练参数
    episodes = 100000
    rewards = []

    # 用于跟踪最优动作和最优奖励
    best_reward = -np.inf  # 初始化为负无穷大，确保任何实际奖励都会比它大
    best_action = None     # 初始化为 None

    # 使用 tqdm 显示进度条
    for e in tqdm(range(episodes), desc="Training Progress"):
        # 重置环境，获取初始观测
        state = env.reset()

        # Agent 选择动作
        action = agent.select_action()

        # 在环境中执行动作，得到奖励、下一个状态和 done 标志
        next_state, reward, done, _ = env.step(action)

        # 更新 Agent 的奖励和选择次数
        agent.update(action, reward)

        # 累积奖励
        rewards.append(reward)

        # 判断是否当前动作是最优动作
        if reward > best_reward:
            best_reward = reward
            best_action = action  # 保存当前最优动作

        # 打印每一轮的结果
        print(f"Episode {e}/{episodes}, Reward: {reward:.2f}, Best Reward: {best_reward:.2f}")

    # 保存最优动作到文件，或在此处使用最优动作
    np.save('best_action.npy', best_action)  # 将最优动作保存到文件
    print("最优动作已保存至 best_action.npy")

    # 绘制训练奖励曲线
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('UCB Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
