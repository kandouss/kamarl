import torch
import numpy as np
import gym
import inspect


def find_cuda_device(device_name):
    cuda_devices = {torch.cuda.get_device_name(f'cuda:{x}'):x for x in range(torch.cuda.device_count())}
    matching_cuda_devices = [x for x in cuda_devices.keys() if device_name in x]

    if len(matching_cuda_devices) == 0:
        return torch.device('cpu')
    return torch.device(f'cuda:{cuda_devices[matching_cuda_devices[0]]}')

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