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

run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")

env = gym.make('Breakout-v0')
save_root = f'/fast/atari_ppo_test/{run_time}'

env = GridRecorder(
    env,
    max_steps=10001,
    save_root=save_root,
    auto_save_interval=500
)

agent = PPOAgent(env.observation_space, env.action_space,
    grid_mode=False,
    hyperparams = {
        'learning_rate': 3.e-4,
        'num_minibatches': 100,
        "minibatch_size": 512,
        "episodes_per_batch": 25,
        'entropy_bonus_coef': 0.0,
        "max_episode_length": 10000,

        'module_hyperparams': {
            'conv_layers': [
                    {'out_channels': 1, 'kernel_size': 3, 'stride': 3, 'padding': 0},
                    {'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 0},
                    {'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 0},
                    {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 0},
            ],
            'input_trunk_layers': [256],
            'lstm_hidden_size': 256,
            'val_mlp_layers': [128],
            'pi_mlp_layers': [128],
        }
    }
)

device = find_cuda_device('1080 Ti')[0]
# device = find_cuda_device('1070')

print(count_parameters(agent.ac))


# wbl = WandbLogger(name='atari_test', project='atari_ppo_test')
# agent.set_logger(wbl)

agent.set_device(device)
import pdb; pdb.set_trace()
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
