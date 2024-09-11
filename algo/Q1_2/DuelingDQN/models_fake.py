import numpy as np
import torch
import torch.nn as nn


class EnhancedDuelingCropQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EnhancedDuelingCropQNetwork, self).__init__()

        self.input_dim = np.prod(input_shape)  # 680 * 4 = 2720
        self.output_shape = output_shape  # 680 * 7 = 4760

        # 增加一个额外的全连接层来增强复杂度
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)  # 新增加的层

        # Value and Advantage streams
        self.fc_value = nn.Linear(512, 256)
        self.fc_advantage = nn.Linear(512, 256)

        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, np.prod(output_shape))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # 通过增加的隐藏层进行处理

        value = torch.relu(self.fc_value(x))
        advantage = torch.relu(self.fc_advantage(x))

        value = self.value(value)
        advantage = self.advantage(advantage)

        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values.reshape(x.size(0), *self.output_shape)

        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values.reshape(x.size(0), *self.output_shape)





class AttentionLayer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.linear = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Linear transformation to embed dimension
        x = self.linear(x)
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x, x, x)
        # Add & norm layer for residual connection
        x = self.norm(attn_output + x)
        return x


class DuelingCropQNetworkWithAttention(nn.Module):
    def __init__(self, input_shape, output_shape, embed_dim=128, num_heads=4):
        super(DuelingCropQNetworkWithAttention, self).__init__()

        self.input_dim = np.prod(input_shape)  # Flattened input dimension (680 * 4 = 2720)
        self.output_shape = output_shape  # Action space dimensions (680 * 7 = 4760)

        # Attention mechanism
        self.attention_layer = AttentionLayer(input_dim=input_shape[1], embed_dim=embed_dim, num_heads=num_heads)

        # Flatten the input for the fully connected layers
        self.fc1 = nn.Linear(embed_dim * input_shape[0], 512)

        # Value and Advantage streams
        self.fc_value = nn.Linear(512, 256)
        self.fc_advantage = nn.Linear(512, 256)

        self.value = nn.Linear(256, 1)  # Value stream
        self.advantage = nn.Linear(256, np.prod(output_shape))  # Advantage stream

    def forward(self, x):
        # Apply attention mechanism on the input features
        x = self.attention_layer(x)

        # Flatten the input for the fully connected layers
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))

        # Value and Advantage streams
        value = torch.relu(self.fc_value(x))
        advantage = torch.relu(self.fc_advantage(x))

        value = self.value(value)
        advantage = self.advantage(advantage)

        # Combine value and advantage streams
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        # Reshape to match the output space
        return q_values.reshape(x.size(0), *self.output_shape)