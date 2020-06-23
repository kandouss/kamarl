import torch
import numpy as np
import torch.nn as nn

import datetime, time
import types

from kamarl.ppo2 import PPOAgent
from kamarl.utils import find_cuda_device, count_parameters
from kamarl.agents import IndependentAgents
from kamarl.logging import WandbLogger

from marlgrid import envs as marl_envs
from marlgrid.utils.video import GridRecorder


run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")
device = find_cuda_device('1080 Ti')[1]

agent_config = {
    'view_tile_size': 3,
    'view_size': 7,
    'observation_style': 'rich',
    'prestige_beta': 3.0, # determines the number of rewards to go from red to blue.
    'hyperparams': {

        "batch_size": 16,
        'num_minibatches': 100,
        "minibatch_size": 256,
        "minibatch_seq_len": 8,
        "hidden_update_interval": 5,

        'learning_rate': 1.e-4, # 1.e-3, #
        "kl_target":  0.01,
        "clamp_ratio": 0.2,
        "lambda":0.97,
        "gamma": 0.99,
        'entropy_bonus_coef': 0.0,#0001,
        'value_loss_coef': 1.0,

        "module_hyperparams": {
            "conv_layers" : [
                {'out_channels': 16, 'kernel_size': 3, 'stride': 3, 'padding': 0},
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ],
            'input_trunk_layers': [128],
            'lstm_hidden_size': 128,
            'val_mlp_layers': [64],
            'pi_mlp_layers': [64]
        }
    }
}

save_root = f'/fast/marlgrid_ppo/{run_time}'

agents = IndependentAgents(
    PPOAgent(**agent_config, color='prestige'),
    # PPOAgent(**agent_config, color='prestige'),
    # PPOAgent(**agent_config, color='prestige'),
)
agents.set_device(device)
print(f"Agents have {count_parameters(agents.agents[0].ac)} parameters.")

# # Params for cluttered multigrid
# env_config = {
#     'grid_size': 9,
#     'max_steps': 100,
#     'seed': 1,
#     'randomize_goal': True,
#     'clutter_density': 0.25,
#     'respawn': True,
#     'ghost_mode': True,
#     'reward_decay': False,
# }
# env = marl_envs.ClutteredMultiGrid([agent.obj for agent in agents], **env_config)

# # Params for goal cycle gridworld
env_config = {
    'grid_size': 9,
    'max_steps': 150,
    'clutter_density': 0.2,
    'seed': np.random.randint(1337*1337),
    'n_bonus_tiles': 3,
    'initial_reward': True,
    'penalty': -0.5,
    'respawn': True,
    'done_condition': 'all',
    'ghost_mode': True,
    'reward_decay': False, # default true.
}
env = marl_envs.ClutteredGoalCycleEnv([agent.obj for agent in agents], **env_config)


wbl = WandbLogger(name='ppo', project='marlgrid_stale_refreshing_test')
agents.set_logger(wbl)
wbl.log_hyperparams({
    'env_name': env.__class__.__name__,
    'env_params': env_config,
    'hparams': agents[0].hyperparams})


# env = GridRecorder(
#     env,
#     max_steps=env_config['max_steps']+1,
#     save_root=save_root,
#     auto_save_interval=100
# )



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
            # env.render(show_agent_views=True)
            agent_total_rewards = None
            while not done:
                # Get an action for each agent.
                # action_array = [agent.action_space.sample() for agent in agents]
                action_array = agents.action_step(obs_array)

                next_obs_array, reward_array, done_array, _ = env.step(action_array)


                total_reward += reward_array.sum()
                ep_reward += reward_array.sum()

                agents.save_step(obs_array, action_array, reward_array, done_array)

                obs_array = next_obs_array
                done = all(done_array) if not np.isscalar(done_array) else done_array

                ep_steps += 1
                total_steps += 1
                # env.render(show_agent_views=True)
                # input()


            ep_time = time.time() - ep_start_time
            if ep_num % 500 == 0:
                agents[0].save(f"/fast/multigrid3/run_{run_time}/episode_{ep_num}/")

            print(
                f"Episode {ep_num: >5d}: len={ep_steps: <4d} | cum rew={total_reward: <4.1f} ({ep_reward: >+3.1f}) | fps={ep_steps/ep_time: >5.2f}"
            )
                
