import torch
import numpy as np
import gym
import inspect
import itertools
import numba
import copy

from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


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

class Collater:
    def __init__(self, space):
        self.space = space
        self.spacelist = self.find_spaces(space)

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
            try:
                for k in ks[:-1]:
                    srcs = [x[k] for x in srcs]
                    tgt = tgt[k]
            except:
                import pdb; pdb.set_trace()
            tgt[ks[-1]] = np.stack([x[ks[-1]] for x in srcs])
        return out

    def decollate(self, item, chunksize):
        tmp, (addr, _) = item, self.spacelist[0]
        for k in addr:
            tmp = tmp[k]
        n_total = len(tmp)
        n_chunks = n_total // chunksize

        retlist = [self.new_empty() for _ in range(n_chunks)]
        for addr, _ in self.spacelist:
            rl_tmp = retlist
            src = item
            for k in addr[:-1]:
                rl_tmp = [x[k] for x in rl_tmp]
                src = src[k]
            for chunk_no, tgt in enumerate(rl_tmp):
                tgt[addr[-1]] = src[addr[-1]][chunk_no*chunksize:(chunk_no+1)*chunksize]

        return retlist

            


# test_space = gym.spaces.Dict({'hi': gym.spaces.Discrete(5)})
# test_space = gym.spaces.Dict({
#     'hi': gym.spaces.Discrete(5),
#     'mom': gym.spaces.Tuple([gym.spaces.Box(0,1,shape=()), gym.spaces.Box(0,1,(2,))]),
# })
# colin = Collater(test_space)
# items = [
#     {'hi': 2, 'mom': [0, [1,2]]},
#     {'hi': 3, 'mom': [2, [3,9]]},
#     {'hi': 4, 'mom': [4, [7,8]]}
# ]
# print(colin.collate(items))
    
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
        return list(zip(*[env.step(action) for env, action in zip(self.envs, actions)]))

    def render(self, *args, which=0, **kwargs):
        if which is None:
            return [e.render(*args, **kwargs) for e in self.envs]
        else:
            return self.envs[which].render(*args, **kwargs)

    def reset(self):
        # self._reset_count = getattr(self, '_reset_count', 0) + 1
        # print(f" rst {self._reset_count}")
        # if self._reset_count > 10:
        #     print("\n"*10)
        #     for line in traceback.format_stack():
        #         print(inspect.stack())
        #     print("\n"*10)
        #     exit()
        
        return [e.reset() for e in self.envs]

    def close(self):
        return

def combine_spaces(spaces):
    return gym.spaces.Tuple(tuple(spaces))

class MultiParallelWrapper(gym.Wrapper):
    def __init__(self, env, env_chunk_size = None):
        if not hasattr(env, 'reward_range'):
            env.reward_range = None
        if not hasattr(env, 'metadata'):
            env.metadata = {}
        super().__init__(env)
        self.action_collater = Collater(env.action_space)
        self.obs_collater = Collater(env.observation_space)
        self.env_chunk_size = env_chunk_size

    def fix_obs(self, obs):
        # import pdb; pdb.set_trace()
        if self.env_chunk_size is not None:
            obs = [x for o in obs for x in o]
        return self.obs_collater.collate(obs)
        # return [self.obs_collater.collate(o) for o in obs]
        
    def render(self, which=0, **kwargs):
        if hasattr(self.env, 'remotes'): # if env is a subprocvecenv, without importing that class
            self.env.remotes[which].send(('render', {'which':which,**kwargs}))
            return self.env.remotes[which].recv()
        elif hasattr(self.env, 'envs'):
            return self.env.envs[which].render(**kwargs)
        else:
            return self.env.render(**kwargs)

    def fix_action(self, action):
        ret = self.action_collater.decollate(action, chunksize=1)
        N = len(ret)
        chunksize = self.env_chunk_size
        return [ret[chunksize*k:chunksize*(k+1)] for k in range(N//chunksize)]

    def fix_scalar(self, item):
        if self.env_chunk_size is not None:
            ret = np.array([x for o in item for x in o])
        else:
            ret = np.array(item)
        return ret.T

    def step(self, action):
        o, r, d, i = self.env.step(self.fix_action(action))
        # import pdb; pdb.set_trace()
        return self.fix_obs(o), self.fix_scalar(r), self.fix_scalar(d), i
    
    def reset(self, **kwargs):
        return self.fix_obs(self.env.reset(**kwargs))

def stack_environments(env_fns, n_subprocs):
    if len(env_fns) % n_subprocs != 0:
        raise ValueError("number of environments should be divisible by number of subprocesses.")
    # env_fns = [env_fns[k::n_subprocs] for k in range(n_subprocs)]
    # tmp = [env_fns[n_subprocs*k:n_subprocs*(k+1):] for k in range(len(env_fns)//n_subprocs)]
    # import pdb; pdb.set_trace()
    chunksize = len(env_fns)//n_subprocs
    env_fns_ = [(lambda: DumberVecEnv(env_fns[chunksize*k:chunksize*(k+1):])) for k in range(n_subprocs)]
    # import pdb; pdb.set_trace()
    # ret = SubprocVecEnv(env_fns)
    # import pdb; pdb.set_trace()
    return MultiParallelWrapper(SubprocVecEnv(env_fns_), env_chunk_size=chunksize)
    