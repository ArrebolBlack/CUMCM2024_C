import numpy as np
import torch
import torch.nn as nn


# 定义CNN Q网络
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_shape[1] * input_shape[2], 512)
        self.fc2 = nn.Linear(512, np.prod(output_shape))  # 输出为 3 维拼接的形状

        # 保存输出形状，用于后续 reshape
        self.output_shape = output_shape

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, *self.output_shape)  # 重塑为输出形状


# 定义更强大的 CNN Q 网络
class EnhancedQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EnhancedQNetwork, self).__init__()

        # 卷积层，深度更大
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # 增加 dropout 层，防止过拟合
        self.dropout = nn.Dropout(p=0.5)

        # 全连接层，尺寸更大
        self.fc1 = nn.Linear(128 * input_shape[1] * input_shape[2], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, np.prod(output_shape))  # 输出为 3 维拼接的形状

        # 保存输出形状，用于后续 reshape
        self.output_shape = output_shape

    def forward(self, state):
        # 前向传播时，依次通过卷积层和全连接层
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平

        # 使用 dropout 防止过拟合
        x = self.dropout(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x.view(-1, *self.output_shape)  # 重塑为输出形状
