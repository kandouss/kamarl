import torch
import numpy as np
import gym
import inspect
import itertools
import numba
from baselines.common.vec_env import VecEnv


@numba.jit#(numba.float32[:](numba.float32[:], numba.float32))
def discount_rewards(rewards, gamma):
    discounted_rewards = 0*rewards
    c0 = 0.0
    ix = len(rewards)-1
    for x in rewards[::-1]:
        c0 = x + gamma * c0
        discounted_rewards[ix] = c0
        ix -= 1
    return discounted_rewards


def count_parameters(mod):
    return np.sum([np.prod(x.shape) for x in mod.parameters()])
    
def find_cuda_device(device_name=''):
    cuda_devices = [torch.device(f'cuda:{x}') for x in range(torch.cuda.device_count())]
    matching_cuda_devices = [dev for dev in cuda_devices if (device_name.lower() in torch.cuda.get_device_name(dev).lower())]
    return matching_cuda_devices
    

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def simplify_box_bounds(value):
    if value is None:
        return None
    elif isinstance(value, np.ndarray):
        first_value = next(value.flat).item()
        if (value==first_value).all():
            return first_value
        else:
            return value
    else:
        raise ValueError(f"Can't interpret box bound of type {type(value)} \n  > {value}")

def space_to_dict(space):
    space_name = space.__class__.__name__
    if isinstance(space, gym.spaces.Dict):
        space_kwargs = {k: space_to_dict(subspace) for k,subspace in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        space_kwargs = {'spaces':[space_to_dict(subspace) for subspace in space.spaces]}
    elif isinstance(space, gym.spaces.Box):
        space_kwargs = {
            'low': simplify_box_bounds(space.low),
            'high': simplify_box_bounds(space.high),
            'shape': tuple(space.shape),
            'dtype': space.dtype.__str__(),
        }
    elif isinstance(space, gym.spaces.Discrete):
        space_kwargs = {
            'n': int(space.n)
        }
    else:
        raise ValueError(f"Can't dict-encode gym space of type {space_name}")

    return {'type': space_name, 'kwargs':space_kwargs}
    
def dict_to_space(enc_space):
    kwargs = enc_space['kwargs']
    space_name = enc_space['type']
    if space_name == 'Dict':
        kwargs = {k: dict_to_space(v) for k,v in kwargs.items()}
    elif space_name == 'Tuple':
        kwargs = {'spaces': dict_to_space(v) for v in kwargs['spaces']}    
    return getattr(gym.spaces, space_name)(**kwargs)

def get_module_inputs(observation_space):
    '''
    minigrid and marlgrid commonly return inputs that are either images or dictionaries.
    This function returns the shapes of image inputs and the total number of scalar inputs from either input style.
    '''
    if isinstance(observation_space, gym.spaces.Box):
        image_shape = observation_space.shape
        n_flat_inputs = 0
    elif isinstance(observation_space, gym.spaces.Dict):
        image_shape = observation_space['pov'].shape
        n_flat_inputs = gym.spaces.flatdim(gym.spaces.Tuple([v for k,v in observation_space.spaces.items() if k != 'pov']))
    else:
        raise ValueError(f"Can't figure out image/flat input sizes for space {observation_space}.")
    return image_shape, n_flat_inputs

    
class DumberVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_spaces = self.observation_space.spaces if isinstance(self.observation_space, gym.spaces.Tuple) else (self.observation_space,)
        self.buf_obs = []
        self.buf_dones = []
        self.buf_rews  = []
        self.buf_infos = []
        self.actions = []

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, *args, **kwargs):
        return [e.render(*args, **kwargs) for e in self.envs]

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        return list(zip(*[e.step(a) for e, a in zip(self.envs, self.actions)]))
        # for i in range(self.num_envs):
        #     obs, rew, done, info = self.envs[i].step(self.actions[i])
        # return self.buf_obs, self.buf_rews, self.buf_dones, self.buf_infos

    def reset(self):
        return [e.reset() for e in self.envs]
        # for i in range(self.num_envs):
        #     obs_tuple = self.envs[i].reset()
        #     if isinstance(obs_tuple, (tuple, list)):
        #         for t,x in enumerate(obs_tuple):
        #             self.buf_obs[t][i] = x
        #     else:
        #         self.buf_obs[0][i] = obs_tuple
        # return self.buf_obs

    def close(self):
        return

def combine_spaces(spaces):
    # if all(isinstance(space, gym.spaces.Discrete) for space in spaces):
    #     return gym.spaces.MultiDiscrete([space.n for space in spaces])
    return gym.spaces.Tuple(tuple(spaces))


class MultiParallelWrapper(gym.Wrapper):
    def __init__(self, env, n_envs, n_agents):
        if not hasattr(env, 'reward_range'):
            env.reward_range = None
        if not hasattr(env, 'metadata'):
            env.metadata = {}
        super().__init__(env)
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.num_envs = n_envs

    def collate(self, agent_obs):
        assert len(agent_obs) == self.n_envs
        if isinstance(agent_obs[0], dict):
            agent_obs = {k: np.stack([o[k] for o in agent_obs]) for k in agent_obs[0].keys()}
        return agent_obs
        

    def fix_obs(self, obs):
        if self.n_agents == len(obs[0]) and self.n_envs == len(obs):
            obs = np.swapaxes(obs, 0, 1)
        return [self.collate(o) for o in obs]
        
        # print("Fixing obs.")
        # import pdb; pdb.set_trace()
        # return obs
    def render(self, which=0, **kwargs):
        if hasattr(self.env, 'remotes'): # if env is a subprocvecenv, without importing that class
            self.env.remotes[which].send(('render', kwargs))
            return self.env.remotes[which].recv()
        elif hasattr(self.env, 'envs'):
            return self.env.envs[which].render(**kwargs)
        else:
            return self.env.render(**kwargs)



    def fix_action(self, action):
        # print(f"ACTION SHAPE IS {np.array(action).shape}")
        # return action
        action = np.array(action)
        if self.n_agents == action.shape[0] and self.n_envs == action.shape[1]:
            return np.swapaxes(action, 0, 1)
        return action
    def fix_scalar(self, item):
        item = np.array(item)
        if len(item.shape)>1:
            if self.n_agents != item.shape[0]:
                return np.swapaxes(item, 0, 1)
        else:
            item = item[None,:]
        return item
    def step(self, action):
        o, r, d, i = self.env.step(self.fix_action(action))
        # import pdb; pdb.set_trace()
        return self.fix_obs(o), self.fix_scalar(r), self.fix_scalar(d), i
    
    def reset(self, **kwargs):
        return self.fix_obs(self.env.reset(**kwargs))