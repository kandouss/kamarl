import torch
import numpy as np
import gym
import inspect

def count_parameters(mod):
    return np.sum([np.prod(x.shape) for x in mod.parameters()])
    
def find_cuda_device(device_name=''):
    cuda_devices = [torch.device(f'cuda:{x}') for x in range(torch.cuda.device_count())]
    matching_cuda_devices = [dev for dev in cuda_devices if (device_name.lower() in torch.cuda.get_device_name(dev).lower())]
    return matching_cuda_devices
    
def space_to_dict(space):
    if not isinstance(space, gym.spaces.Space):
        raise ValueError

    space_params = {k:getattr(space, k, v.default) for k,v in inspect.signature(space.__class__).parameters.items()}
    space_name = space.__class__.__name__

    for k in ['low','high']:
        if isinstance(space_params.get(k,None), np.ndarray):
            val = next(space_params[k].flat)
            if (space_params[k] == val).all():
                space_params[k] = val.item()

    if 'dtype' in space_params:
        space_params['dtype'] = space_params['dtype'].__str__()

    return {
        'type': space_name,
        'kwargs': space_params
    }

def dict_to_space(space_dict):
    return getattr(gym.spaces, space_dict['type'])(**space_dict['kwargs'])

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

    def fix_obs(self, obs):
        if self.n_agents == obs.shape[1] and self.n_envs == obs.shape[0]:
            return np.swapaxes(obs, 0, 1)
        return obs

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
        return item
    def step(self, action):
        o, r, d, i = self.env.step(self.fix_action(action))
        # import pdb; pdb.set_trace()
        return self.fix_obs(o), self.fix_scalar(r), self.fix_scalar(d), i
    
    def reset(self, **kwargs):
        return self.fix_obs(self.env.reset(**kwargs))