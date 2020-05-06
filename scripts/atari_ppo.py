import torch
import numpy as np
import torch.nn as nn

import datetime, time

from kamarl.ppo import PPOAgent
from kamarl.utils import find_cuda_device
from marlgrid.utils.video import GridRecorder
import gym


count_parameters = lambda mod: np.sum([np.prod(x.shape) for x in mod.parameters()])

env = gym.make('Breakout-v0')

env = GridRecorder(env, max_len=10001)

agent = PPOAgent(env.observation_space, env.action_space, 
        hyperparams = {
            'module_hyperparams': {
                'conv_layers': [
                    {'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                    {'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                    {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                    {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                    {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
            ]}
        })

device = find_cuda_device('1080 Ti')

print(count_parameters(agent.ac))
agent.set_device(device)

run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")

total_reward = 0
num_episodes = int(1e6)
with torch.set_grad_enabled(False):
    for ep_num in range(num_episodes):
        # Initialize the environment and state
        obs = env.reset()
        done = False

        agent.start_episode()

        ep_start_time = time.time()

        env.recording = ep_num % 500 == 0
        ep_steps = 0
        # env.render(show_agent_views=True)
        agent_total_rewards = None
        while not done:

            # Get an action for each agent.
            # action_array = [agent.action_space.sample() for agent]
            act = agent.action_step(obs)

            next_obs, reward, done, _ = env.step(act)

            total_reward += reward

            agent.save_step(obs, act, reward, done)

            obs = next_obs

            ep_steps += 1
            # env.render(show_agent_views=True)

        ep_time = time.time() - ep_start_time
        if env.recording:
            env.export_video(
                f"/fast/multigrid3/run_{run_time}/episode_{ep_num}.mp4",
                render_frame_images=True,
            )


        print(
            f"Episode {ep_num: >5d}: {ep_steps: <4d} ticks | cum rew={total_reward: <4.1f} | fps={ep_steps/ep_time: >5.2f}"
        )

        agent.end_episode()
