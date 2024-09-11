import numpy as np
import torch
import torch.nn as nn
import gym
from env import CropPlantingEnv

# 定义适用于 CropPlantingEnv 的 CNN Q 网络
class CropQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CropQNetwork, self).__init__()
        # 卷积层用于处理输入
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 全连接层
        conv_output_size = 32 * input_shape[1] * input_shape[2]  # 根据卷积后展平的尺寸调整
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, np.prod(output_shape))  # 输出形状与动作空间匹配

        # 保存输出形状，用于重塑
        self.output_shape = output_shape

    def forward(self, state):
        # 应用卷积层
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平

        # 应用全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # 重塑输出以匹配动作空间
        return x.view(-1, *self.output_shape)


# 定义适用于 CropPlantingEnv 的增强型 CNN Q 网络，具有更多层和 dropout
class EnhancedCropQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EnhancedCropQNetwork, self).__init__()

        # 更深的卷积层
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # dropout 防止过拟合
        self.dropout = nn.Dropout(p=0.5)

        # 全连接层
        conv_output_size = 128 * input_shape[1] * input_shape[2]  # 根据卷积后展平的尺寸调整
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, np.prod(output_shape))  # 输出形状与动作空间匹配

        # 保存输出形状，用于重塑
        self.output_shape = output_shape

    def forward(self, state):
        # 应用更深的卷积层
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平

        # 使用 dropout 防止过拟合
        x = self.dropout(x)

        # 应用全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # 重塑输出以匹配动作空间
        return x.view(-1, *self.output_shape)


# 示例用法
env = CropPlantingEnv()
observation_shape = env.observation_space.shape
action_shape = env.action_space.shape
print("obs_space:", observation_shape)
print("act_space:", action_shape)


# 初始化网络
q_network = CropQNetwork(observation_shape, action_shape)
enhanced_q_network = EnhancedCropQNetwork(observation_shape, action_shape)

# 示例观测和前向传播
sample_obs = torch.FloatTensor(env.reset()).unsqueeze(0)
output = q_network(sample_obs)
enhanced_output = enhanced_q_network(sample_obs)

print(output.shape)  # 应该匹配动作空间的形状
print(enhanced_output.shape)
