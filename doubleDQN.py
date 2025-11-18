#Double DQN

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Neural Network 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32).to(device),
            torch.tensor(action).to(device),
            torch.tensor(reward).to(device),
            torch.tensor(next_state, dtype=torch.float32).to(device),
            torch.tensor(done, dtype=torch.float32).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# Policy
def select_action(state, policy_net, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()

# Training Loop
def train_dqn(
    env,
    episodes=1000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    learning_rate=1e-3,
    target_update=10,
    batch_size=128
):
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer()

    epsilon = epsilon_start
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset(seed=SEED)
        total_reward = 0

        while True:
            action = select_action(state, policy_net, epsilon, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                s_batch, a_batch, r_batch, ns_batch, d_batch = replay_buffer.sample(batch_size)

                #Q(s,a) από policy_net
                q_values = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

                #Double DQN target
                with torch.no_grad():
                    next_actions = policy_net(ns_batch).argmax(1)  # Eπιλογή δράσης με policy_net
                    next_q_values = target_net(ns_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # Aξιολόγηση με target_net

                target_q = r_batch + gamma * next_q_values * (1 - d_batch)

                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return rewards_per_episode

# Plotting
def plot_rewards(rewards, window_size=10):
    mean_rewards = plot_rewards_mean(rewards, window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", color="skyblue", alpha=0.4)
    plt.plot(mean_rewards, label=f"Moving Average (window={window_size})", color="darkblue", linewidth=2)
    plt.title("Double DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("double_dqn_mountaincar_rewards.png")
    plt.show()

def plot_rewards_mean(rewards, d):
    m = []
    for i in range(len(rewards)):
        if i < d:
            m.append(np.mean(rewards[:i+1]))
        else:
            m.append(np.mean(rewards[i-d+1:i+1]))
    return m



if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.action_space.seed(SEED)
    rewards = train_dqn(env)
    plot_rewards(rewards, window_size=10)
    env.close()