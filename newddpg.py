import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
import matplotlib.pyplot as plt
from collections import deque
import argparse

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return (torch.tensor(state, dtype=torch.float32, device=device),
                torch.tensor(action, dtype=torch.float32, device=device),
                torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1),
                torch.tensor(next_state, dtype=torch.float32, device=device))

    def __len__(self):
        return len(self.buffer)

# === LSTM World Model ===
class LSTMStatePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)

# === Reward Functions ===
def reward_fast_descent_v1(z, z_vel, z_min, z_max, zvel_min, zvel_max):
    norm_z = (z - z_min) / (z_max - z_min + 1e-8)
    norm_zvel = (z_vel - zvel_min) / (zvel_max - zvel_min + 1e-8)
    return -norm_z + (1 - norm_zvel)

def reward_fast_descent_v2(z, z_vel, z_min, z_max, zvel_min, zvel_max):
    norm_z = (z - z_min) / (z_max - z_min + 1e-8)
    norm_zvel = (z_vel - zvel_min) / (zvel_max - zvel_min + 1e-8)
    return -norm_z + 2 * (1 - norm_zvel)**2

def reward_fast_descent_v3(z, z_vel, z_min, z_max, zvel_min, zvel_max):
    ideal_vel_low = -100
    ideal_vel_high = -50
    if z_vel < ideal_vel_low:
        vel_reward = (z_vel - ideal_vel_low) / abs(ideal_vel_low)
    elif z_vel > ideal_vel_high:
        vel_reward = (ideal_vel_high - z_vel) / abs(ideal_vel_high)
    else:
        vel_reward = 1.0
    vel_reward = max(vel_reward, 0)
    norm_z = (z - z_min) / (z_max - z_min + 1e-8)
    return vel_reward - norm_z

# === Environment ===
class WorldModelEnv:
    def __init__(self, model_path, device, reward_fn, folder="normalized_npz", seq_len=100):
        self.device = device
        self.seq_len = seq_len
        self.z_idx = 2
        self.zvel_idx = 12
        self.z_min = -69.39875
        self.z_max = -1.001669
        self.zvel_min = -37.90419769
        self.zvel_max = 39.5221786
        self.reward_fn = reward_fn

        self.model = LSTMStatePredictor(35, 128, 24).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.folder = folder
        self.reset()

    def reset(self):
        file = random.choice(glob(os.path.join(self.folder, "*.npz")))
        data = np.load(file, allow_pickle=True)
        state = np.array(data["norm_state"][:100], dtype=np.float32)
        action = np.array(data["norm_action"][:99], dtype=np.float32)
        self.state_seq = torch.tensor(state, device=self.device)
        action = torch.tensor(action, device=self.device)
        sa_seq = torch.cat([self.state_seq[:99], action], dim=1)
        self.input_seq = sa_seq.unsqueeze(0)
        self.current_state = self.state_seq[99].unsqueeze(0)
        return self.current_state

    def step(self, action):
        sa_new = torch.cat([self.current_state, action], dim=1)
        input_seq = torch.cat([self.input_seq, sa_new.unsqueeze(1)], dim=1)
        with torch.no_grad():
            next_state = self.model(input_seq)[:, -1, :]
        self.input_seq = input_seq[:, 1:, :]
        self.current_state = next_state.detach()

        z = next_state[0, self.z_idx].item()
        z_vel = next_state[0, self.zvel_idx].item()
        reward = self.reward_fn(z, z_vel, self.z_min, self.z_max, self.zvel_min, self.zvel_max)

        return self.current_state, reward

# === Actor and Critic ===
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        if len(s.shape) == 3:
            s = s.squeeze(1)
        if len(a.shape) == 3:
            a = a.squeeze(1)
        return self.net(torch.cat([s, a], dim=1))

# === Training ===
def train_ddpg(epochs, reward_version):
    reward_map = {
        1: reward_fast_descent_v1,
        2: reward_fast_descent_v2,
        3: reward_fast_descent_v3,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = WorldModelEnv("lstm_model.pt", device, reward_map[reward_version])
    actor = Actor(24, 11).to(device)
    critic = Critic(24, 11).to(device)
    actor_target = Actor(24, 11).to(device)
    critic_target = Critic(24, 11).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    buffer = ReplayBuffer()
    reward_log = []

    gamma = 0.99
    tau = 0.005
    batch_size = 64

    for epoch in range(epochs):
        state = env.reset()
        total_reward = 0
        for t in range(100):
            with torch.no_grad():
                action = actor(state)
            next_state, reward = env.step(action)
            buffer.push(state.cpu().numpy(), action.cpu().numpy(), reward, next_state.cpu().numpy())
            state = next_state.detach()
            total_reward += reward

            if len(buffer) >= batch_size:
                s_batch, a_batch, r_batch, ns_batch = buffer.sample(batch_size, device)

                a2 = actor_target(ns_batch)
                q_target = r_batch + gamma * critic_target(ns_batch, a2)
                q_pred = critic(s_batch, a_batch)
                critic_loss = nn.MSELoss()(q_pred, q_target.detach())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                a_pred = actor(s_batch)
                actor_loss = -critic(s_batch, a_pred).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        reward_log.append(total_reward)
        print(f"Epoch {epoch+1}, Total Reward: {total_reward:.2f}")

    final_reward = reward_log[-1]
    suffix = f"_r{final_reward:.2f}_v{reward_version}"
    torch.save(actor.state_dict(), f"ddpg_actor{suffix}.pt")
    torch.save(critic.state_dict(), f"ddpg_critic{suffix}.pt")
    print(f"\n✅ Model saved: ddpg_actor{suffix}.pt, ddpg_critic{suffix}.pt")

    plt.plot(reward_log, label="Reward", color='blue', alpha=0.5)
    if len(reward_log) >= 10:
        moving_avg = np.convolve(reward_log, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(reward_log)), moving_avg, label="10-Episode Moving Avg", color='orange', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title(f"DDPG with Reward v{reward_version}")
    plt.grid()
    plt.legend()
    plt.savefig(f"ddpg_reward_curve{suffix}.png")
    print(f"📈 Plot saved as ddpg_reward_curve{suffix}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--reward", type=int, default=3, choices=[1,2,3], help="Reward function version (1, 2, or 3)")
    args = parser.parse_args()
    train_ddpg(args.epochs, args.reward)
