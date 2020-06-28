import torch
import os
import numpy as np
import torch.nn as nn

import datetime, time

from kamarl.ppo import PPOAgent
from kamarl.utils import find_cuda_device, count_parameters
from kamarl.logging import WandbLogger

from marlgrid.utils.video import GridRecorder
import gym

class ImageShrinker(gym.ObservationWrapper):
    def __init__(self, env, factor=2):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("Can only shrink (downsample) plain images.")
        old_os = env.observation_space
        self.factor = factor
        self.observation_space = gym.spaces.Box(
            low=np.take(old_os.low, 0),
            high=np.take(old_os.high, 0),
            shape=self.observation(old_os.sample()).shape,
            dtype=old_os.dtype)

    def observation(self, img):
        return img[self.factor//2::self.factor, self.factor//2::self.factor, :]
        # img = img[:self.factor*(img.shape[0]//self.factor), :self.factor*(img.shape[1]//self.factor)]
        # return img.reshape((img.shape[0]//self.factor, self.factor, img.shape[1]//self.factor, self.factor, 3)).mean(axis=(1,3)).astype(self.observation_space.dtype)
    
    def render(self, *args, **kwargs):
        return self.observation(self.env.render(*args, **kwargs))


run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")

env = gym.make('Breakout-v4')
save_root = f'/tmp/atari_ppo_test/{run_time}'

env = ImageShrinker(env, factor=3)

env = GridRecorder(
    env,
    max_steps=10001,
    save_root=save_root,
    auto_save_interval=500
)
ppo_learning_config = {
    "batch_size": 8,
    'num_minibatches': 10,
    "minibatch_size": 256,
    "minibatch_seq_len": 8,
    "hidden_update_interval": 2,

    'learning_rate': 1.e-4, # 1.e-3, #
    "kl_target":  0.01,
    "clamp_ratio": 0.2,
    "lambda":0.97,
    "gamma": 0.99,
    'entropy_bonus_coef': 0.0,#0001,
    'value_loss_coef': 1.0,
}

ppo_model_config = {
    "conv_layers" : [
        {'out_channels': 8, 'kernel_size': 3, 'stride': 3, 'padding': 0},
        {'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 0},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 0},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 0},
    ],
    'input_trunk_layers': [192],
    'lstm_hidden_size': 192,
    'val_mlp_layers': [64,64],
    'pi_mlp_layers': [64,64],
}


agent = PPOAgent(
    observation_space = env.observation_space,
    action_space = env.action_space,
    learning_config = ppo_learning_config,
    model_config = ppo_model_config,
)

# device = find_cuda_device('1080 Ti')[1]
device = torch.device('cpu')

print(f"Agent has {count_parameters(agent.ac)} parameters.")


wbl = WandbLogger(name='atari_test_laptop', project='atari_ppo_test')
agent.set_logger(wbl)

agent.set_device(device)

total_reward = 0
num_episodes = int(1e6)
with torch.set_grad_enabled(False):
    for ep_num in range(num_episodes):
        # Initialize the environment and state
        obs = env.reset()
        done = False

        agent.start_episode()

        ep_start_time = time.time()

        ep_steps = 0
        ep_reward = 0

        agent_total_rewards = None
        while not done:

            # Get an action for each agent.
            act = agent.action_step(obs)

            next_obs, reward, done, _ = env.step(act)

            total_reward += reward
            ep_reward += reward

            agent.save_step(obs, act, reward, done)

            obs = next_obs

            ep_steps += 1
            # env.render(show_agent_views=True)

        ep_time = time.time() - ep_start_time
        if ep_num % 500 == 0:
            agent.save(os.path.join(save_root, f"episode_{ep_num}"))
        print(
            f"Episode {ep_num: >5d}: {ep_steps: <4d} ticks | cum rew={total_reward: <4.1f} ({ep_reward: >+2.1f}) | fps={ep_steps/ep_time: >5.2f}"
        )

        agent.end_episode()
