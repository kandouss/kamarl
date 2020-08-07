import numpy as np
from multiprocessing import Process, Pipe
import gym
from .utils import Collater
import cloudpickle


class MultiParallelWrapper(gym.Wrapper):
    def __init__(self, env, n_envs, env_chunk_size = None):
        if not hasattr(env, 'reward_range'):
            env.reward_range = None
        if not hasattr(env, 'metadata'):
            env.metadata = {}
        super().__init__(env)

        self.action_collater = Collater(env.action_space)
        self.obs_collater = Collater(env.observation_space)
        self.env_chunk_size = env_chunk_size
        self.n_envs = n_envs

    def fix_obs(self, obs):
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
        # Input has dimensions of (n_agents, n_parallel_envs). Transpose to (n_parallel_envs, n_agents.)
        tmp = self.action_collater.decollate(action, n_chunks=self.n_envs)
        return [(tmp[self.env_chunk_size*k:self.env_chunk_size*(k+1)]) for k in range(self.n_envs//self.env_chunk_size)]

    def fix_scalar(self, item):
        if self.env_chunk_size is not None:
            ret = np.array([x for o in item for x in o])
        else:
            ret = np.array(item)
        return ret.T

    def step(self, action):
        obs, rew, done, meta = self.env.step(self.fix_action(action))
        return self.fix_obs(obs), self.fix_scalar(rew), self.fix_scalar(done), meta
    
    def reset(self, **kwargs):
        return self.fix_obs(self.env.reset(**kwargs))


def stack_environments(env_fns, n_subprocs):
    if len(env_fns) % n_subprocs != 0:
        raise ValueError("number of environments should be divisible by number of subprocesses.")
    chunksize = len(env_fns)//n_subprocs

    def get_hooks(k):
        return lambda: EnvStack(env_fns[chunksize*k:chunksize*(k+1)])
    env_fns_ = [get_hooks(k) for k in range(n_subprocs)]

    return MultiParallelWrapper(SubprocEnvStack(env_fns_), n_envs = len(env_fns), env_chunk_size=chunksize)


class EnvStack:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.env_batch_size = len(self.envs)

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, actions):
        return list(zip(*[env.step(action) for env, action in zip(self.envs, actions)]))

    def render(self, *args, which=0, **kwargs):
        if which is None:
            return [e.render(*args, **kwargs) for e in self.envs]
        else:
            return self.envs[which].render(*args, **kwargs)

    def reset(self):
        return [e.reset() for e in self.envs]

    def close(self):
        del self.envs
        return
        


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            remote.send(env.step(data))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'render':
            remote.send(env.render(**data))
        elif cmd == 'getattr':
            remote.send(getattr(env, data))
        else:
            raise NotImplementedError


class SubprocEnvStack:
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()


    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, rews, dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return [remote.recv() for remote in self.remotes]

    def render(self, **kwargs):
        for remote in self.remotes:
            remote.send(('render', kwargs))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def step(self, actions):

        self.step_async(actions)
        return self.step_wait()
