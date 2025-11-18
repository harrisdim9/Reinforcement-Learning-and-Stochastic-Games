# DRQN (LSTM-DQN)


import gymnasium as gym
import numpy as np
import random
import collections
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



class Q_net(nn.Module):
    def __init__(self, state_space=None, action_space=None, hidden_space=128):
        super(Q_net, self).__init__()
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.state_space  = state_space

        self.layer1       = nn.Linear(self.state_space, self.hidden_space)
        self.lstm         = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.layer2       = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x                 = F.relu(self.layer1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))        
        q_seq             = self.layer2(x)               
        return q_seq, new_h, new_c

    @torch.no_grad()
    def sample_action(self, observation, h, c, epsilon, action_space_n):
        if random.random() < epsilon:
            a = random.randrange(action_space_n)
            return a, h, c
        out, new_h, new_c = self.forward(observation, h, c) 
        a = out.squeeze(0).squeeze(0).argmax().item()
        return a, new_h, new_c

    def init_hidden_state(self, batch_size, training=None):
        if training:
            return (torch.zeros(1, batch_size, self.hidden_space, device=device),
                    torch.zeros(1, batch_size, self.hidden_space, device=device))
        else:
            return (torch.zeros(1, 1, self.hidden_space, device=device),
                    torch.zeros(1, 1, self.hidden_space, device=device))

class EpisodeBuffer:
    def __init__(self):
        self.observation = []
        self.action      = []
        self.reward      = []
        self.next_obs    = []
        self.done        = []

    def put(self, transition):
        self.observation.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        observation = np.array(self.observation, dtype=np.float32) 
        action      = np.array(self.action, dtype=np.int64)        
        reward      = np.array(self.reward, dtype=np.float32)      
        next_obs    = np.array(self.next_obs, dtype=np.float32)    
        done        = np.array(self.done, dtype=np.float32)        

        if random_update is True:
            observation = observation[idx:idx+lookup_step]
            action      = action[idx:idx+lookup_step]
            reward      = reward[idx:idx+lookup_step]
            next_obs    = next_obs[idx:idx+lookup_step]
            done        = done[idx:idx+lookup_step]

        return dict(observation=observation, acts=action, rews=reward, next_obs=next_obs, done=done)

    def __len__(self) -> int:
        return len(self.observation)

class EpisodeMemory:
    def __init__(self, random_update=False, max_epi_num=100, max_epi_len=200, batch_size=32, lookup_step=30):
        self.random_update = random_update
        self.max_epi_num   = max_epi_num
        self.max_epi_len   = max_epi_len
        self.batch_size    = batch_size
        self.lookup_step   = lookup_step
        self.memory        = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode: EpisodeBuffer):
        self.memory.append(episode)

    def sample(self):
        assert len(self.memory) > 0
        sampled_buffer = []

        if self.random_update:
            sampled_episodes = random.sample(self.memory, self.batch_size)
            L = self.lookup_step
            for ep in sampled_episodes:
                if len(ep) < L:
                    idx = 0
                    sample = ep.sample(random_update=True, lookup_step=len(ep), idx=idx)
                    pad_len = L - len(ep)
                    for k in ("observation", "acts", "rews", "next_obs", "done"):
                        sample[k] = self._pad_sequence(sample[k], L, key=k)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(ep) - L + 1)
                    sample = ep.sample(random_update=True, lookup_step=L, idx=idx)
                    sampled_buffer.append(sample)
            seq_len = self.lookup_step
        else:
            ep = random.choice(self.memory)
            sample = ep.sample(random_update=False)
            sampled_buffer.append(sample)
            seq_len = len(sample["observation"])

        return sampled_buffer, seq_len

    def _pad_sequence(self, arr, L, key="observation"):
        if arr.ndim == 1:
            out = np.zeros((L,), dtype=arr.dtype)
            out[:len(arr)] = arr
            out[len(arr):] = arr[-1]
            return out
        else:
            d = arr.shape[1]
            out = np.zeros((L, d), dtype=arr.dtype)
            out[:len(arr), :] = arr
            out[len(arr):, :] = arr[-1, :]
            return out

    def __len__(self):
        return len(self.memory)


def get_observation_fully_observable(state):
    return state

def get_observation_pomdp(state):
    return np.array([state[0]], dtype=np.float32)


def get_observation_noisy_pomdp(state, noise_std=0.01):
    noise = np.random.normal(0, noise_std)
    return np.array([state[0] + noise], dtype=np.float32)





def convert_data(data, batch_size, seq_len):
    data = np.array(data, dtype=np.float32)                    # (B, S, d) ή (B,S)
    return torch.tensor(data.reshape(batch_size, seq_len, -1), dtype=torch.float32, device=device)

def compute_rolling_average(rewards, window_size=10):
    return [np.mean(rewards[max(0, i - window_size + 1): i + 1]) for i in range(len(rewards))]

def plot_single_reward(rewards, title, window_size=10):
    avg = compute_rolling_average(rewards, window_size)
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Episode Reward", color="skyblue", alpha=0.35)
    plt.plot(avg, label=f"Moving Avg ({window_size})", color="darkblue", linewidth=2)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_step(q_net, target_q_net, episode_memory, optimizer, gamma=0.99, grad_clip=5.0):
    samples, seq_len = episode_memory.sample()  

    B = len(samples)
    observations      = []
    actions           = []
    rewards           = []
    next_observations = []
    dones             = []

    for i in range(B):
        observations.append(samples[i]["observation"]) 
        actions.append(samples[i]["acts"])             
        rewards.append(samples[i]["rews"])             
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])                



    observations      = convert_data(observations, B, seq_len)       
    rewards           = convert_data(rewards, B, seq_len)           
    next_observations = convert_data(next_observations, B, seq_len)   
    dones             = convert_data(dones, B, seq_len)              

    rewards = rewards.squeeze(-1)  
    dones   = dones.squeeze(-1)     
    actions = np.array(actions, dtype=np.int64).reshape(B, seq_len, 1)
    actions = torch.tensor(actions, dtype=torch.long, device=device) 

    #Target Q
    h_t, c_t  = target_q_net.init_hidden_state(batch_size=B, training=True)
    q_target, _, _  = target_q_net(next_observations, h_t, c_t)      
    next_q_max      = q_target.max(dim=2, keepdim=True)[0]          
    target_q_values = rewards.unsqueeze(-1) + gamma * next_q_max * dones.unsqueeze(-1)

    #Current Q(s,a)
    h, c        = q_net.init_hidden_state(batch_size=B, training=True)
    q_out, _, _ = q_net(observations, h, c)                         
    q_sa        = q_out.gather(2, actions)                           

    loss = F.smooth_l1_loss(q_sa, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()

def train_drqn(env,
               observation_function,
               num_episodes=900,
               max_step=200,
               batch_size=32,
               learning_rate=1e-3,
               gamma=0.99,
               eps_start=1.0,
               eps_end=0.01,
               eps_decay=0.995,
               target_update_frequency=10,
               tau=0.05,
               random_update=True,
               lookup_step=30,
               min_epi_num=32,
               verbose=True):

    input_dim  = env.observation_space.shape[0]
    output_dim = env.action_space.n

    q_net      = Q_net(input_dim, output_dim, hidden_space=128).to(device)
    target_net = Q_net(input_dim, output_dim, hidden_space=128).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    episode_memory = EpisodeMemory(random_update=random_update,
                                   max_epi_num=100,
                                   max_epi_len=max_step,
                                   batch_size=batch_size,
                                   lookup_step=lookup_step)

    epsilon = eps_start
    rewards = []

    for ep in range(1, num_episodes + 1):
        state, _      = env.reset(seed=SEED)
        obs           = observation_function(state).astype(np.float32)
        done          = False
        ep_rew        = 0.0
        ep_buf        = EpisodeBuffer()
        h, c          = q_net.init_hidden_state(batch_size=1, training=False)

        for t in range(max_step):
            obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)  # (1,1,obs_dim)
            action, h, c = q_net.sample_action(obs_t, h, c, epsilon, output_dim)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs = observation_function(next_state).astype(np.float32)

            not_done = 0.0 if done else 1.0
          
            ep_buf.put([obs, action, reward / 100.0, next_obs, not_done])

            obs    = next_obs
            ep_rew += reward
            if done:
                break



            if len(episode_memory) >= min_epi_num and (t + 1) % target_update_frequency == 0:
                train_step(q_net, target_net, episode_memory, optimizer, gamma=gamma)
              

                with torch.no_grad():
                    for p_t, p in zip(target_net.parameters(), q_net.parameters()):
                        p_t.data.copy_(tau * p.data + (1.0 - tau) * p_t.data)

       
        epsilon = max(eps_end, epsilon * eps_decay)

        episode_memory.put(ep_buf)
        rewards.append(ep_rew)

        if verbose and ep % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(f"Episode {ep}, Reward: {avg:.2f}, Epsilon: {epsilon:.3f}")

    return rewards


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.action_space.seed(SEED)

    rewards = train_drqn(
        env=env,
        observation_function=get_observation_fully_observable,  
        num_episodes=900,
        max_step=200,
        batch_size=32,
        learning_rate=1e-3,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        target_update_frequency=10,
        tau=0.05,
        random_update=True,
        lookup_step=30,
        min_epi_num=32,
        verbose=True
    )


    plt.figure(figsize=(10, 5))
    avg = compute_rolling_average(rewards, window_size=10)
    plt.plot(rewards, label="Episode Reward", color="skyblue", alpha=0.4)
    plt.plot(avg, label="Moving Avg (window=10)", color="darkblue", linewidth=2)
    plt.title("DRQN (LSTM-DQN) on MountainCar-v0")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    env.close()