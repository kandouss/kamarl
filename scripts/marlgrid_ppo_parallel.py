import torch
import numpy as np
import torch.nn as nn

import datetime, time
import os

from kamarl.ppo import PPOAgent
from kamarl.utils import find_cuda_device, count_parameters, MultiParallelWrapper, DumberVecEnv    
from kamarl.agents import IndependentAgents
from kamarl.logging import WandbLogger

from marlgrid.envs import env_from_config
# from marlgrid import envs as marl_envs
from marlgrid.agents import GridAgentInterface
from marlgrid.utils.video import GridRecorder

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")
device = find_cuda_device('1080 Ti')[1]
# device = torch.device('cpu')

save_root = os.path.abspath(os.path.expanduser(f'/tmp/marlgrid_ppo_refactor/{run_time}'))

experiment_config = {
    'n_parallel_envs': 16, # set this to None in order to disable subprocess environment vectorization.
    'total_episodes': int(1e6),
    'checkpoint_interval': 8192/2, # episodes.
    'recording_interval': 100,
    'save_root': save_root
}

# Config for a cluttered multigrid
env_config = {
    'env_class': 'ClutteredMultiGrid',

    'grid_size': 9,
    'max_steps': 150,
    'clutter_density': 0.2,
    'respawn': True,
    'done_condition': 'all',
    'ghost_mode': True,
    'reward_decay': False, # default true.
}

# Config for a cluttered goal cycle environment
env_config = {
    'env_class': 'ClutteredGoalCycleEnv',
    'grid_size': 9,
    'max_steps': 150,
    'clutter_density': 0.2,
    'respawn': True,
    'done_condition': 'all',
    'ghost_mode': True,
    'reward_decay': False, # default true.
    'n_bonus_tiles': 3,
    'initial_reward': True,
    'penalty': -0.5,
}

agent_interface_config = {
    'view_tile_size': 3,
    'view_size': 9,
    'view_offset': 3,
    'observation_style': 'rich',
    'prestige_beta': 0.95, # determines the rate at which prestige decays
    'color': 'prestige',
}

ppo_learning_config = {
    "batch_size": 32,
    'num_minibatches': 30,
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
        {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
    ],
    'input_trunk_layers': [192],
    'lstm_hidden_size': 192,
    'val_mlp_layers': [64,64],
    'pi_mlp_layers': [64,64],
}

load_agents = [] # List of save paths of already-made agents to load into the env
n_new_agents = 2 # Number of new agents to be created with the above config/hyperparameters.


grid_agents = []
new_agents_info = [
    {'interface_config': agent_interface_config, 'learning_config': ppo_learning_config, 'model_config': ppo_model_config}
    for _ in range(n_new_agents)
]

grid_agents = []

for agent_load_path in load_agents:
    grid_agents.append(PPOAgent.load(agent_load_path))


for agent_info in new_agents_info:
    iface = GridAgentInterface(**agent_info['interface_config'])
    new_fella = PPOAgent(
        observation_space=iface.observation_space,
        action_space=iface.action_space, 
        learning_config=agent_info['learning_config'],
        model_config=agent_info['model_config'],
    )
    new_fella.metadata['marlgrid_interface'] = agent_interface_config
    grid_agents.append(new_fella)


agents = IndependentAgents(*grid_agents)

agents.set_device(device)
print(f"Agents have {count_parameters(agents.agents[0].ac)} parameters.")


# Quick save/load test
agents[0].save('/tmp/agent_save_test', force=True)
loaded_agent = PPOAgent.load('/tmp/agent_save_test')


assert agents[0] is not loaded_agent
for mod_name in ['ac']:
    saved_weights = getattr(agents[0], mod_name).state_dict()
    loaded_weights = getattr(loaded_agent, mod_name).state_dict()
    for k,v in saved_weights.items():
        w1 = v.cpu()
        w2 = loaded_weights[k].cpu()
        assert((w1==w2).all())
agents.agents[0] = loaded_agent


def make_environment(agents, experiment_config, env_config, seed_bump=0):
    env_config['agents'] = [agent.metadata['marlgrid_interface'] for agent in agents]

    n_par = experiment_config['n_parallel_envs']
    if n_par is None:
        env = env_from_config(env_config)
    else:
        env_hooks = [(lambda: env_from_config({**env_config, 'seed': k})) for k in range(n_par)]
        env = MultiParallelWrapper(DumberVecEnv(env_hooks), n_agents=len(agents), n_envs=n_par)
        # env = MultiParallelWrapper(SubprocVecEnv(env_hooks), n_agents=len(agents), n_envs=n_par)

    if experiment_config['recording_interval'] is not None:
        return GridRecorder(env, save_root=experiment_config['save_root'], max_steps=env_config['max_steps']+1)
    else:
        return env


env = make_environment(
    agents,
    experiment_config=experiment_config, 
    env_config=env_config)


wbl = None
# wbl = WandbLogger(name='ppo', project='marlgrid_stale_refreshing_test')
# agents.set_logger(wbl)
# wbl.log_hyperparams({
#     'env_name': env.__class__.__name__,
#     'env_params': env_config,
#     'hparams': agents[0].hyperparams})

last_recorded = -experiment_config['recording_interval']-1

n_parallel = experiment_config['n_parallel_envs'] or 1


total_reward = 0
total_steps = 0
ep_num = 0

while ep_num < experiment_config['total_episodes']:
    # Initialize the environment and state
    obs_array = env.reset()
    done = False
    agents.start_episode(n_parallel=experiment_config['n_parallel_envs'])
    with torch.set_grad_enabled(False):
        ep_start_time = time.time()

        ep_steps = 0
        ep_reward = 0
        
        if isinstance(env, GridRecorder):
            env.recording = ep_num - last_recorded > experiment_config['recording_interval']

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
            done = np.array(done_array).all()

            ep_steps += 1
            total_steps += 1
            

        ep_time = time.time() - ep_start_time
        if ep_num % 500 == 0:
            agents[0].save(os.path.join(save_root, "checkpoints", f"episode_{ep_num}/"))

        print(
            f"Episode {ep_num: >5d}: len={ep_steps: <4d} | cum rew={total_reward: <4.1f} ({ep_reward: >+3.1f}) | fps={ep_steps*n_parallel/ep_time: >5.2f}"
        )

    agents.end_episode()

    if isinstance(env, GridRecorder) and env.recording:
        env.export_both(save_root=os.path.join(save_root, 'recordings'), ident=f'episode_{ep_num}')
        last_recorded = ep_num
        env.recording = False
    
    ep_num += n_parallel