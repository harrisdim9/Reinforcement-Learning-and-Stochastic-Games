#ppo
#!pip install stable-baselines3[extra]

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback



SEED = 42
np.random.seed(SEED)


#Callback για rewards ανά επεισόδιο 
class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done and "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
        return True


env = gym.make("MountainCar-v0")
env.reset(seed=SEED)
env.action_space.seed(SEED)
env = Monitor(env)


vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

callback = RewardCallback()



model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=0,
    seed=SEED,
    learning_rate=5e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.9,
    gamma=0.99,
    n_epochs=20,
    clip_range=0.2,
  
)


model.learn(total_timesteps=130_000, callback=callback)

# Plot rewards + moving average
rewards = callback.episode_rewards
window_size = 10
moving_avg = [np.mean(rewards[max(0, i - window_size + 1):i + 1]) for i in range(len(rewards))]

plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Episode Reward", alpha=0.4)
plt.plot(moving_avg, label=f"Moving Avg (window={window_size})", color='darkblue')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Training on MountainCar-v0")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ppo_mountaincar_rewards.png")
plt.show()