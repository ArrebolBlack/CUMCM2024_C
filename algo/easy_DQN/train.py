import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 引入进度条
from collections import deque
from dqn import CropQNetwork, EnhancedCropQNetwork

import sys
sys.path.append('/home/shengjiaao/Newpython/C_2024')
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告
from env import CropPlantingEnv

# DQN Agent
class DQNAgent:
    def __init__(self, input_shape, output_shape, use_enhanced=True, device=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.batch_size = 512
        self.target_update_freq = 5  # 更新目标网络的频率
        # self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda')


        # 选择使用普通网络还是增强网络
        self.model = EnhancedCropQNetwork(input_shape, output_shape).to(self.device) if use_enhanced else CropQNetwork(
            input_shape, output_shape).to(self.device)
        self.target_model = EnhancedCropQNetwork(input_shape, output_shape).to(self.device) if use_enhanced else CropQNetwork(
            input_shape, output_shape).to(self.device)

        # 使用 DataParallel 以支持多GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.target_model = nn.DataParallel(self.target_model)

        self.update_target_model()  # 初始化目标模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()  # 损失函数

    def update_target_model(self):
        """更新目标网络的权重"""
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        """根据当前状态选择动作"""
        if np.random.rand() <= self.epsilon:
            # 选择随机动作
            return np.random.uniform(0, 1, self.output_shape)  # 返回符合动作空间的随机动作

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 加入 batch 维度，并移动到 device
        print(state.shape)
        act_values = self.model(state)

        # 确保去掉批量维度
        if act_values.shape[0] == 1:
            act_values = act_values.squeeze(0)

        # return act_values.cpu().detach().numpy().squeeze()  # 返回动作（3D 矩阵）
        return act_values.cpu().detach().numpy()  # 返回动作（不需要 squeeze）

    def remember(self, state, action, reward, next_state, done):
        """将经验存入记忆库"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """从记忆库中采样并训练模型"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if not done:
                with torch.no_grad():  # 禁用梯度计算
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            # 计算目标 Q 值
            # target_f = self.model(state).clone()
            # target_f[0] = torch.FloatTensor(action).to(self.device)  # 更新动作矩阵的 Q 值
            target_f = self.model(state).clone()
            # 使用目标网络计算下一步动作的Q值
            target_f[0] = target  # 更新目标动作的 Q 值

            states.append(state)
            targets_f.append(target_f)

        # 批量训练
        states = torch.cat(states)
        targets_f = torch.cat(targets_f)

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(states), targets_f)
        loss.backward()
        self.optimizer.step()

        # 按比例衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()  # 返回 loss 值用于记录

    def load(self, name):
        """加载训练好的模型"""
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        """保存训练好的模型"""
        torch.save(self.model.state_dict(), name)


# 主训练过程
if __name__ == "__main__":

    env = CropPlantingEnv()
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    agent = DQNAgent(observation_shape, action_shape)

    # 训练参数
    episodes = 5000
    done = False
    rewards = []
    losses = []
    best_reward = -float('inf')
    save_model_path = 'best_dqn_model.pth'

    # 使用 tqdm 显示进度条
    for e in tqdm(range(episodes), desc="Training Progress"):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0

        for time in range(1000):
            action = agent.act(state)  # 动作是输出的 3D 矩阵
            # print("action.shape", action.shape)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # 训练并获取 loss
            loss = agent.replay()
            if loss is not None:
                episode_loss += loss


            if done:
                break

        # 更新 target 网络
        if e % agent.target_update_freq == 0:
            agent.update_target_model()

        # 保存训练奖励和 loss
        rewards.append(episode_reward)
        losses.append(episode_loss)

        # 打印每一轮的训练结果
        print(
            f"Episode: {e}/{episodes}, Reward: {episode_reward:.2f}, Loss: {episode_loss:.4f}, Epsilon: {agent.epsilon:.2f}")

        # 保存最优模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(save_model_path)
            print(f"Best model saved with reward: {best_reward:.2f}")

    # 绘制训练奖励和 loss 曲线
    plt.figure(figsize=(12, 6))

    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 绘制 loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()
