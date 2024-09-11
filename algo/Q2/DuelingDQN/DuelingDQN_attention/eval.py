import torch
import numpy as np
import os
from env_new import CropPlantingEnv
from models import EnhancedDuelingCropQNetwork


class DQNAgent:
    def __init__(self, input_shape, output_shape, model_path='best_dqn_model.pth', device=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = EnhancedDuelingCropQNetwork(input_shape, output_shape).to(self.device)

        # 处理 DataParallel 的模型
        state_dict = torch.load(model_path)

        # 如果模型是在 DataParallel 模式下保存的，state_dict 的键会带有 "module." 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # 去掉 "module." 前缀
            new_state_dict[new_key] = value

        # 加载修正后的 state_dict
        self.model.load_state_dict(new_state_dict)

        self.model.eval()  # 设置模型为推理模式

    def act(self, state):
        """根据当前状态选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return act_values.cpu().detach().numpy().squeeze(0)

if __name__ == "__main__":
    env = CropPlantingEnv()
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    # 允许用户指定模型路径，默认为 'best_dqn_model.pth'
    model_path = '/home/ziwu/Newpython/C_2024/algo/Q2/DuelingDQN/DuelingDQN_attention/checkpoints'
    agent = DQNAgent(observation_shape, action_shape, model_path=model_path)

    state = env.reset()
    done = False
    best_reward = -float('inf')
    best_action = None

    # 推理若干步并保存最优动作
    for step in range(1000):
        action = agent.act(state)  # 选择动作
        next_state, reward, done, _ = env.step(action)

        # 更新状态
        state = next_state

        # 如果得到更高的 reward，则保存该动作
        if reward > best_reward:
            best_reward = reward
            best_action = action

        if done:
            break

    # 保存最优动作为 .npy 文件
    save_action_dir = '/home/ziwu/Newpython/C_2024/algo/Q2/DuelingDQN/DuelingDQN_attention/result'
    save_action_path = os.path.join(save_action_dir, 'best_action_ddqn_attn_2.npy')

    # 确保目录存在
    os.makedirs(save_action_dir, exist_ok=True)

    if best_action is not None:
        np.save(save_action_path, best_action)
        print(f"Best action saved with reward: {best_reward:.2f} at {save_action_path}")
    else:
        print("No action found to be saved.")
