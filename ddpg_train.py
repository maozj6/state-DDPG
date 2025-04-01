import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import numpy as np
from scipy.signal import savgol_filter


def compute_reward_savgol(data, digit_number, window_length=21, polyorder=3):
    z_velocity = data[0, :, digit_number].cpu().numpy()  # shape: (150,)

    # 仅对前149帧进行平滑
    z_prev = z_velocity[:149]

    # 防止窗口太大或小
    if window_length >= 149:
        window_length = 149 if 149 % 2 == 1 else 148
    if window_length < polyorder + 2:
        window_length = polyorder + 2
    if window_length % 2 == 0:
        window_length += 1  # 必须是奇数

    # 应用 Savitzky-Golay 滤波器
    z_smooth = savgol_filter(z_prev, window_length, polyorder)

    # 使用多项式外推预测第150个值
    x = np.arange(149)
    coeffs = np.polyfit(x, z_smooth, polyorder)
    z_pred = np.polyval(coeffs, 149)

    z_actual = z_velocity[149]
    diff = np.abs(z_actual - z_pred)

    reward = np.exp(-10 * diff ** 2)
    return float(reward)
class LSTMModel(nn.Module):
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, output_size=17):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len=150, input_size=22)
        out, (hn, cn) = self.lstm(x)  # out: (batch, seq_len, hidden)
        final_hidden = hn[-1]        # take last layer's hidden state
        output = self.fc(final_hidden)
        return output


def compute_reward(data, digit_number):
    """
    data: Tensor of shape (1, 150, 17)
    digit_number: int, which digit to extract from last dimension
    """
    z_velocity = data[0, :, digit_number]  # shape: (150,)

    # 前149帧
    z_prev = z_velocity[:149]

    # 用线性拟合前149个点
    x = torch.arange(149, dtype=torch.float32)
    y = z_prev

    A = torch.stack([x, torch.ones_like(x)], dim=1)  # [149, 2]
    coeffs, _ = torch.lstsq(y.unsqueeze(1), A)  # shape: [2, 1]
    k, b = coeffs[:2].squeeze()

    # 预测第150个点
    z_pred = k * 149 + b
    z_actual = z_velocity[149]

    # 误差越小，reward 越高，可以用负MSE或者高斯函数
    mse = F.mse_loss(z_pred, z_actual)

    reward = torch.exp(-10 * mse)  # 越接近1越好，越远reward越低

    return reward.item()


def compute_reward_diff(data, digit_number):
    z_velocity = data[0, :, digit_number]  # shape: (150,)
    deltas = z_velocity[1:150] - z_velocity[0:149]  # shape: (149,)

    mean_delta = deltas[:-1].mean()  # 前148个点的平均增量
    last_delta = deltas[-1]  # 第149->150帧的增量

    diff = torch.abs(last_delta - mean_delta)

    reward = torch.exp(-10 * diff ** 2)
    return reward.item()


# 假设你已经定义了这两个函数
# from your_env_module import env, reward_function

# 环境输入输出维度
STATE_DIM = 17
ACTION_DIM = 5

# 超参数
HIDDEN_DIM = 256
ACT_LIMIT = 1.0
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR = 1e-3
MAX_EPISODES = 500
MAX_STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 网络
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * ACT_LIMIT

# Critic 网络
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2 = map(np.stack, zip(*batch))
        return map(lambda x: torch.tensor(x, dtype=torch.float32).to(device), (s, a, r, s2))

    def __len__(self):
        return len(self.buffer)

def main():


    env = LSTMModel()


    actor = Actor().to(device)
    target_actor = Actor().to(device)
    critic = Critic().to(device)
    target_critic = Critic().to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR)
    replay_buffer = ReplayBuffer()

    def train():
        if len(replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state = replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            next_action = target_actor(next_state)
            target_q = reward.unsqueeze(1) + GAMMA * target_critic(next_state, next_action)

        q_value = critic(state, action)
        critic_loss = nn.MSELoss()(q_value, target_q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = -critic(state, actor(state)).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 软更新
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    # 主循环
    for episode in range(MAX_EPISODES):
        state = 
        #load an inital state here
        episode_reward = 0

        for step in range(MAX_STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = actor(state_tensor).squeeze(0).detach().cpu().numpy()
            action += np.random.normal(0, 0.1, size=ACTION_DIM)  # 探索噪声
            #suppose next_state has a size of (1,length,17) , 17 is the state size
            next_state = env(state, action)
            reward = reward_function(next_state,digit_number)

            replay_buffer.push(state, action, reward, next_state)
            train()

            state = next_state
            episode_reward += reward

        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    main()
