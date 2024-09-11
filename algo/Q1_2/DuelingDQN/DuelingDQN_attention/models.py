import torch
import torch.nn as nn
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention_output, _ = self.attention(query, key, value)
        x = self.dropout(self.norm1(attention_output + query))
        forward_output = self.feed_forward(x)
        out = self.dropout(self.norm2(forward_output + x))
        return out

class EnhancedDuelingCropQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, embed_size=512, heads=8, dropout=0.1, forward_expansion=2):
        super(EnhancedDuelingCropQNetwork, self).__init__()

        self.input_dim = np.prod(input_shape)
        self.output_shape = output_shape

        # 输入层
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, embed_size)

        # Transformer 模块
        self.transformer = TransformerBlock(embed_size, heads, dropout, forward_expansion)

        # 值函数和优势函数的分支
        self.fc_value = nn.Linear(embed_size, 256)
        self.fc_advantage = nn.Linear(embed_size, 256)

        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, np.prod(output_shape))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # 将输入展平成 (batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # 这是 Transformer 的输入嵌入

        # 调整输入以适配 Transformer 输入格式 (批次大小, 序列长度, 嵌入维度)
        x = x.unsqueeze(1)  # 添加序列维度，假设我们这里只有一个"序列"

        # Transformer 需要输入格式为 (序列长度, 批次大小, 嵌入维度)
        x = x.permute(1, 0, 2)  # 变换为 (序列长度, 批次大小, 嵌入维度)
        transformer_output = self.transformer(x, x, x)

        # Transformer 处理后恢复原始格式 (批次大小, 嵌入维度)
        x = transformer_output.permute(1, 0, 2).squeeze(1)

        # 值函数和优势函数分支
        value = torch.relu(self.fc_value(x))
        advantage = torch.relu(self.fc_advantage(x))

        value = self.value(value)
        advantage = self.advantage(advantage)

        # 计算 Q 值，Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values.reshape(x.size(0), *self.output_shape)



if __name__ == "__main__":
    # 定义模型输入和输出的形状
    input_shape = (1082,26)  # 输入序列长度为 35000
    output_shape = (1082,7)  # 输出序列长度为 8000

    # 创建模型实例
    model = EnhancedDuelingCropQNetwork(input_shape, output_shape, embed_size=512, heads=8, dropout=0.1,
                                        forward_expansion=2)

    # 打印模型结构，检查定义是否正确
    print("Model Structure:\n", model)

    # 生成一个随机的输入数据，假设 batch_size 为 4
    batch_size = 4
    random_input = torch.randn(batch_size, *input_shape)

    # 执行前向传播，检查模型是否能够处理输入
    try:
        output = model(random_input)
        print(f"Model output shape: {output.shape}")
        assert output.shape == (batch_size, *output_shape), "输出形状不匹配！"
        print("模型定义正确，前向传播成功！")
    except Exception as e:
        print(f"模型定义或前向传播过程中出错: {e}")