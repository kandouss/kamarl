import torch
import numpy as np
import gym
import inspect
import itertools
import numba
import copy
import collections.abc

def pd_mode(pd):
    if isinstance(pd, torch.distributions.Categorical):
        return torch.argmax(pd.logits, axis=-1)
    else:
        raise NotImplementedError

class log_lists:
    def __init__(self):
        self.list_dict = {}

    def update(self, d):
        for k, v in d.items():
            self.log_value(k, v)

    def __getitem__(self, key):
        if key in self.list_dict:
            return self.list_dict[key]
        else:
            return []

    def np(self, key):
        return np.array(self.list_dict[key])
    
    def mean(self, key):
        return np.nanmean(self.list_dict[key], axis=0)

    def var(self, key):
        return np.nanvar(self.list_dict[key], axis=0)

    def std(self, key):
        return np.nanstd(self.list_dict[key], axis=0)

    def log_value(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim==0):
            value = value.item()
        self.list_dict.setdefault(key, []).append(value)
        
def params_grads(mod):
    params = {k: v.detach() for k, v in mod.named_parameters()}
    grads = {k: v.grad if v.grad is not None else torch.zeros_like(params[k]) for k,v in mod.named_parameters()}
    return params, grads


@numba.njit(numba.float32[:](numba.float32[:], numba.float32))
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    c0 = 0.0
    for ix in range(len(rewards)-1, -1, -1):
        c0 = rewards[ix] + gamma * c0
        discounted_rewards[ix] = c0
    return discounted_rewards

def discount_rewards_tensor(rewards, gamma):
    return torch.from_numpy(discount_rewards(rewards.cpu().numpy(), gamma)).to(rewards.device)

def count_parameters(mod):
    return np.sum([np.prod(x.shape) for x in mod.parameters()])
    
def find_cuda_device(device_name=''):
    cuda_devices = [torch.device(f'cuda:{x}') for x in range(torch.cuda.device_count())]
    matching_cuda_devices = [dev for dev in cuda_devices if (device_name.lower() in torch.cuda.get_device_name(dev).lower())]
    return matching_cuda_devices
    
def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def update_config_dict(base_config, new_config):
    novel_keys = []
    updated_config = copy.deepcopy(base_config)
    for k,v in new_config.items():
        if k not in base_config:
            novel_keys.append(k)
        updated_config[k] = v
    return updated_config, novel_keys

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


def combine_spaces(spaces):
    return gym.spaces.Tuple(spaces)

def get_slices(n_total, n_chunks):
    lens = [0 for _ in range(n_chunks)]
    for k in range(n_total):
        lens[k%n_chunks] += 1
    slices = []
    count = 0
    for chunk_len in lens:
        slices.append(slice(count, count+chunk_len))
        count += chunk_len
    return slices

class Collater:
    def __init__(self, space):
        self.space = space
        self.spacelist = self.find_spaces(space)
        self.slice_cache = {}

    @classmethod
    def find_spaces(cls, space, keys=tuple()):
        if isinstance(space, gym.spaces.Dict):
            return [x for k,v in space.spaces.items() for x in cls.find_spaces(space=v, keys=(*keys, k))]
        elif isinstance(space, gym.spaces.Tuple):
            return [x for k,v in enumerate(space.spaces) for x in cls.find_spaces(space=v, keys=(*keys, k))]
        else:
            return [((*keys,), space)]

    def new_empty(self, space=None):
        if space is None:
            space = self.space
        if isinstance(space, gym.spaces.Dict):
            return {k: self.new_empty(space=v) for k,v in space.spaces.items()}
        elif isinstance(space, gym.spaces.Tuple):
            return [self.new_empty(space=v) for v in space.spaces]
        else:
            return None
    
    def collate(self, items):
        if not isinstance(self.space, (gym.spaces.Dict, gym.spaces.Tuple)):
            assert isinstance(items, np.ndarray)
            return np.stack(items)
        out = self.new_empty()
        for ks, space in self.spacelist:
            srcs = items
            tgt = out
            # try:
            for k in ks[:-1]:
                srcs = [x[k] for x in srcs]
                tgt = tgt[k]
            # except:
            #     import pdb; pdb.set_trace()
            # try:
            tgt[ks[-1]] = np.stack([x[ks[-1]] for x in srcs])
            # except:
            #     import pdb; pdb.set_trace()
        return out

    def get_slices(self, n_total, n_chunks):
        if n_chunks is None:
            return None
        if (n_total, n_chunks) in self.slice_cache:
            return self.slice_cache[(n_total, n_chunks)]
        else:
            ret = get_slices(n_total, n_chunks)
            self.slice_cache[(n_total, n_chunks)] = ret
            return ret

    def decollate(self, items, n_chunks):
        slices = None
        collated_output = [self.new_empty() for _ in range(n_chunks)]
        for addr, _ in self.spacelist:
            out_ptrs = collated_output
            src_ptr = items

            for k in addr[:-1]:
                out_ptrs = [out_ptr[k] for out_ptr in out_ptrs]
                src_ptr = src_ptr[k]

            src_vals = src_ptr[addr[-1]]
            chunksize = len(src_vals)//n_chunks
            
            for out_ptr, slc in zip(out_ptrs, self.get_slices(len(src_vals), n_chunks)):
                out_ptr[addr[-1]] = src_vals[slc]

        return collated_output