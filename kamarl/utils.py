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

    def collate(self, agent_obs):
        assert len(agent_obs) == self.n_envs
        if isinstance(agent_obs[0], dict):
            agent_obs = {k: np.stack([o[k] for o in agent_obs]) for k in agent_obs[0].keys()}
        return agent_obs
        

    def fix_obs(self, obs):
        if self.n_agents == obs.shape[1] and self.n_envs == obs.shape[0]:
            obs = np.swapaxes(obs, 0, 1)
        return [self.collate(o) for o in obs]
        
        # print("Fixing obs.")
        # import pdb; pdb.set_trace()
        # return obs

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