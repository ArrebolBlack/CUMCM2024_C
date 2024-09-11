import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from env_new import CropPlantingEnv
import numpy as np


# Transformer Encoder-Decoder 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, heads, num_layers, forward_expansion, dropout):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(embed_size, heads, forward_expansion, dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.fc_out = nn.Linear(embed_size, output_dim)
        self.embedding = nn.Linear(input_dim, embed_size)
        self.target_embedding = nn.Linear(output_dim, embed_size)

    def forward(self, src, trg):
        # Embedding for input and target sequences
        src_embed = self.embedding(src)
        trg_embed = self.target_embedding(trg)

        # Transformer Encoder-Decoder structure
        enc_output = self.encoder(src_embed)
        dec_output = self.decoder(trg_embed, enc_output)

        # Final output projection
        out = self.fc_out(dec_output)
        return out

# PPO 算法
class PPOAgent:
    def __init__(self, env, input_dim, output_dim, embed_size=512, heads=8, num_layers=6, forward_expansion=4, dropout=0.1, lr=3e-4, gamma=0.99, eps_clip=0.2, entropy_coef=0.05):
        self.env = env
        self.policy = nn.DataParallel(TransformerModel(input_dim, output_dim, embed_size, heads, num_layers, forward_expansion, dropout)).to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef  # 熵系数
        self.policy_old = nn.DataParallel(TransformerModel(input_dim, output_dim, embed_size, heads, num_layers, forward_expansion, dropout)).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state, trg):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        trg = torch.FloatTensor(trg).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.policy_old(state, trg)
        dist = torch.sigmoid(action_probs)  # 将动作输出映射到0-1之间
        action = dist.squeeze(0)  # 去除batch维度，使action的形状为 (1082, 7)
        entropy = -(dist * torch.log(dist + 1e-10)).sum()  # 计算熵
        return action.cpu().numpy(), dist, entropy  # 返回动作、其分布和熵

    def optimize(self, memory):
        for state, action, reward, done, trg, entropy in memory:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            trg = torch.FloatTensor(trg).unsqueeze(0).to(device)

            # 获取新的策略网络输出
            action_probs = self.policy(state, trg)
            new_action = torch.sigmoid(action_probs)

            # 计算 advantage
            action = torch.FloatTensor(action).to(device)  # 确保 action 在 GPU 上
            advantage = reward + (1 - done) * self.gamma * new_action - action  # 保持 advantage 计算仍在 GPU 上

            # PPO 损失计算
            ratio = torch.exp(new_action - action)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            # 添加熵正则化项
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())


import matplotlib.pyplot as plt

def train():
    env = CropPlantingEnv()
    agent = PPOAgent(env, input_dim=26, output_dim=7)
    num_episodes = 200
    best_reward = -float('inf')  # 用于记录最高的reward
    best_action = None  # 保存最佳的动作
    reward_history = []  # 用于保存每个episode的总reward

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        memory = []
        target_seq = np.random.randn(1082, 7)  # 动作的目标序列

        for t in range(200):  # 限制每个episode的最大步数
            action, _, entropy = agent.select_action(state, target_seq)  # 增加对熵值的接收
            next_state, reward, done, _ = env.step(action)


            memory.append([state, action, reward, done, target_seq, entropy])  # 保存熵值
            state = next_state
            total_reward += reward

            if done:
                break

        # 保存最佳模型和动作
        if total_reward > best_reward:
            best_reward = total_reward
            best_action = action
            # 保存模型
            torch.save(agent.policy.state_dict(), 'best_model.pth')
            print(f'Saved best model at episode {episode} with reward {best_reward}')

        reward_history.append(total_reward)  # 记录每个episode的总reward
        agent.optimize(memory)

        print(f'Episode {episode}, Total Reward: {total_reward}')

    # 保存最佳动作
    np.save('best_action_1.npy', best_action)
    print('Training complete. Best model and action saved.')

    # 绘制reward历史图
    plot_reward_curve(reward_history)

# 绘制reward随时间的变化图
def plot_reward_curve(reward_history):
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.savefig('reward_curve.png')  # 保存图像
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()