import torch
import numpy as np
import torch.nn as nn

import datetime, time
import types

from kamarl.ppo import PPOAgent
from kamarl.utils import find_cuda_device, count_parameters
from kamarl.agents import IndependentAgents
from kamarl.logging import WandbLogger

from marlgrid import envs as marl_envs
from marlgrid.utils.video import GridRecorder


run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")
device = find_cuda_device('1080 Ti')

marlgrid_agent_kwargs = {
    'view_tile_size': 3,
    'view_size': 7,
}
save_root = f'/fast/atari_ppo_test/{run_time}'

agents = IndependentAgents(
    PPOAgent(**marlgrid_agent_kwargs, color='red'),
    PPOAgent(**marlgrid_agent_kwargs, color='blue'),
    PPOAgent(**marlgrid_agent_kwargs, color='purple')
)
agents.set_device(device)
print(count_parameters(agents.agents[0].ac))

grid_params = {
    'grid_size': 10,
    'max_steps': 100,
    'seed': 1,
    'randomize_goal': True,
    'clutter_density': 0.2,
    'respawn': True,
    'ghost_mode': True,
}
env = marl_envs.ClutteredMultiGrid([agent.obj for agent in agents], **grid_params)


wbl = WandbLogger(name='ppo', project='marlgrid_test')
agents.set_logger(wbl)
wbl.log_hyperparams({
    'env_name': env.__class__.__name__,
    'env_params': grid_params,
    'hparams': agents[0].hyperparams})


env = GridRecorder(
    env,
    max_steps=grid_params['max_steps']+1,
    save_root=save_root,
    auto_save_interval=500
)



total_reward = 0
total_steps = 0
num_episodes = int(1e6)
for ep_num in range(num_episodes):
    # Initialize the environment and state
    obs_array = env.reset()
    done = False
    with agents.episode():
        with torch.set_grad_enabled(False):
            ep_start_time = time.time()

            ep_steps = 0
            ep_reward = 0
            env.render(show_agent_views=True)
            agent_total_rewards = None
            while not done:
                # Get an action for each agent.
                # action_array = [agent.action_space.sample() for agent]
                action_array = agents.action_step(obs_array)

                next_obs_array, reward_array, done_array, _ = env.step(action_array)

                done = all(done_array)

                total_reward += reward_array.sum()
                ep_reward += reward_array.sum()

                agents.save_step(obs_array, action_array, reward_array, done_array)

                obs_array = next_obs_array

                ep_steps += 1
                total_steps += 1
                # input()
                env.render(show_agent_views=True)

                if wbl is not None:
                    wbl.flush_values(step=True)


            ep_time = time.time() - ep_start_time
            if ep_num % 500 == 0:
                agents[0].save(f"/fast/multigrid3/run_{run_time}/episode_{ep_num}/")

            print(
                f"Episode {ep_num: >5d}: len={ep_steps: <4d} | cum rew={total_reward: <4.1f} ({ep_reward: >+3.1f}) | fps={ep_steps/ep_time: >5.2f}"
            )
                
