import torch
import numpy as np
import torch.nn as nn
import gym

import datetime, time
import os

from kamarl.ppo_rec import PPOAEAgent
from kamarl.utils import find_cuda_device, count_parameters
from kamarl.vec_env import stack_environments
from kamarl.agents import IndependentAgents
from kamarl.log import WandbLogger

from marlgrid.envs import env_from_config
# from marlgrid import envs as marl_envs
from marlgrid.agents import GridAgentInterface
from marlgrid.utils.video import GridRecorder

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")

device = find_cuda_device('1080 Ti')[1]

save_root = os.path.abspath(os.path.expanduser(f'/fast/marlgrid_ppo_parallel/{run_time}'))


class ConsistencyTestEnv(gym.Env):
    observation_space_ = gym.spaces.Dict(
            {'pov': gym.spaces.Box(low=0.0, high=1.0, shape=(15,15,3), dtype=np.float32)}
        )
    action_space_ = gym.spaces.Discrete(10)
    def __init__(self, n_agents, **config):


        self.n_agents = n_agents
        self.config = config
        self.n_episodes = 0
        self.n_steps = 0
        self.action_space = gym.spaces.Tuple([self.action_space_ for _ in range(self.n_agents)])
        self.observation_space = gym.spaces.Tuple([self.observation_space_ for _ in range(self.n_agents)])
        # super().__init__()

    def reset(self):
        self.n_episodes += 1
        self.n_steps = 0
        return self.get_obs()

    @property
    def seed(self):
        return self.config.get('seed', 0)

    @property
    def max_steps(self):
        return self.config.get('max_steps', 100)

    def get_obs(self, vals=None):
        # povs = [np.full(shape=self.observation_space['pov'].shape, fill_value=self.n_steps/self.max_steps + k/10.)
        #     for k in range(self.n_agents)]
        # return {'pov': np.stack(povs)}
        if vals is None:
            return [
                {'pov': np.full(shape=self.observation_space[k]['pov'].shape, fill_value=self.n_steps/self.max_steps + k/10., dtype=np.float32)}
                for k in range(self.n_agents)
            ]
        else:
            return [
                {'pov': np.full(shape=self.observation_space[k]['pov'].shape, fill_value=vals[k], dtype=np.float32)}
                for k in range(self.n_agents)
            ]

    def get_rew(self):
        return [self.seed for k in range(self.n_agents)]

    def step(self, actions):
        self.n_steps += 1
        done = self.n_steps > self.max_steps
        return (
            self.get_obs(actions),
            self.get_rew(),
            done,
            {}
        )
    
    def render(self, *args, **kwargs):
        print("Env render!")


experiment_config = {
    'n_parallel_envs': 56, # set this to None in order to disable subprocess environment vectorization.
    'n_subprocs': 8,
    'total_episodes': int(1e6),
    'checkpoint_interval': 8192/2, # episodes.
    'recording_interval': 1000,
    'save_root': save_root
}
env_config = {
    'max_steps': 100
}


ppo_learning_config = {
    "batch_size": 64,
    'num_minibatches': 20,
    "minibatch_size": 512,
    "minibatch_seq_len": 16,
    "hidden_update_interval": 2,
    "hidden_update_n_parallel": 64,

    'learning_rate': 1.e-4, # 1.e-3, #
    "kl_target":  0.01,
    "clamp_ratio": 0.2,
    "lambda":0.97,
    "gamma": 0.993,
    'entropy_bonus_coef': 0.003,#0001,
    'value_loss_coef': 0.05,
    'reconstruction_loss_coef': 0.5,
    'reconstruction_loss_loss': 'l1',

    'save_test_image': os.path.join(save_root,'sample_reconstruction.png')
}


ppo_model_config = {
    "conv_layers" : [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 3, 'padding': 0},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        # {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
    ],
    'input_trunk_layers': [128],
    'lstm_hidden_size': 192,
    'val_mlp_layers': [64,32],
    'pi_mlp_layers': [64,32],
}

n_new_agents = 3 # Number of new agents to be created with the above config/hyperparameters.

grid_agents = []
new_agents_info = [
    {'learning_config': ppo_learning_config, 'model_config': ppo_model_config}
    for _ in range(n_new_agents)
]

grid_agents = []

for agent_info in new_agents_info:
    new_fella = PPOAEAgent(
        observation_space=ConsistencyTestEnv.observation_space_,
        action_space=ConsistencyTestEnv.action_space_, 
        learning_config=agent_info['learning_config'],
        model_config=agent_info['model_config'],
    )
    grid_agents.append(new_fella)


agents = IndependentAgents(*grid_agents)

agents.set_device(device)
print(f"Agents have {count_parameters(agents.agents[0].ac)} parameters.")


# Quick save/load test
# agents[0].save('/tmp/agent_save_test', force=True)
# loaded_agent = PPOAgent.load('/tmp/agent_save_test')


# assert agents[0] is not loaded_agent
# for mod_name in ['ac']:
#     saved_weights = getattr(agents[0], mod_name).state_dict()
#     loaded_weights = getattr(loaded_agent, mod_name).state_dict()
#     for k,v in saved_weights.items():
#         w1 = v.cpu()
#         w2 = loaded_weights[k].cpu()
#         assert((w1==w2).all())
# agents.agents[0] = loaded_agent


def make_environment(agents, experiment_config, env_config, seed_bump=0):
    # env_config['agents'] = [agent.metadata['marlgrid_interface'] for agent in agents]

    n_par = experiment_config['n_parallel_envs']

    def make_hook(k):
        new_config = {**env_config, 'seed': k}
        return lambda: ConsistencyTestEnv(n_agents = len(grid_agents), seed=k)
    env_hooks = [make_hook(k) for k in range(n_par)]
    env = stack_environments(env_hooks, n_subprocs=experiment_config['n_subprocs'])
        
    return env
    # if experiment_config['recording_interval'] is not None:
    #     return GridRecorder(env, save_root=experiment_config['save_root'], max_steps=env_config['max_steps']+1)
    # else:
    #     return env


env = make_environment(
    agents,
    experiment_config=experiment_config, 
    env_config=env_config)


wbl = None

print()
print(f"Saving in {save_root}")

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
    action_record = []
    obs_record = []
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
            # action_array = agents.action_step(obs_array)
            
            action_array = np.array([k for k in range(len(obs_array))])
            action_array = [
                k*100 + np.arange(n_parallel)
                for k in range(len(agents.agents))
            ]
            action_array = np.array([env.action_space.sample() for _ in range(n_parallel)]).T
            # import pdb; pdb.set_trace()
            action_record.append(action_array)
            obs_record.append(obs_array)

            next_obs_array, reward_array, done_array, _ = env.step(action_array)

            # import pdb; pdb.set_trace()
            total_reward += reward_array.sum()
            ep_reward += reward_array.sum()

            agents.save_step(obs_array, action_array, reward_array, done_array)

            obs_array = next_obs_array
            done = np.array(done_array).any()
            # print(done)

            ep_steps += 1
            total_steps += 1
            

        ep_time = time.time() - ep_start_time
        if ep_num % 500 == 0:
            agents[0].save(os.path.join(save_root, "checkpoints", f"episode_{ep_num}/"))
        n_agents = len(agents.agents)
        print(
            f"Episode {ep_num: >5d}: len={ep_steps: <4d} | cum rew={total_reward: <4.1f} ({ep_reward/n_parallel/n_agents: >+3.1f}) | fps={ep_steps*n_parallel/ep_time: >5.2f}"
        )

    agents.end_episode()
    ar_ref = np.array(action_record).transpose(1,2,0)
    ar_rec = np.array([[agent.replay_memory.episodes[k].act for k in range(len(agent.replay_memory.episodes))] for agent in agents.agents])
    rew_rec = np.array([[agent.replay_memory.episodes[k].rew for k in range(len(agent.replay_memory.episodes))] for agent in agents.agents])
    o1 = np.array([obs_step[1]['pov'][...,0,0,0] for obs_step in obs_record])
    o2 = np.array([ep['obs']['pov'][...,0,0,0] for ep in agents[1].replay_memory.episodes])
    check_ = lambda agent_no, ep_no: agents[agent_no].replay_memory.episodes[ep_no]['obs']['pov'][1:,...,0,0,0] == agents[agent_no].replay_memory.episodes[ep_no]['act'][:-1]

    # testf = lambda n, k: agents.agents[n].replay_memory.episodes[k].act
    # rpm = agents.agents[0].replay_memory.episodes[0]
    # rpm2 = agents.agents[1].replay_memory.episodes[0]
    # tmp = np.array([testf(0,k)[:3] for k in range(10)])
    import pdb; pdb.set_trace()

    if isinstance(env, GridRecorder) and env.recording:
        env.export_both(save_root=os.path.join(save_root, 'recordings'), episode_id=f'episode_{ep_num}')
        last_recorded = ep_num
        env.recording = False
    
    ep_num += n_parallel
