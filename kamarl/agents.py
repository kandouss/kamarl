import os, sys
import json
import torch
import numpy as np
import types
import gym
from abc import ABC, abstractmethod, abstractproperty
from kamarl.utils import space_to_dict, dict_to_space, combine_spaces
from marlgrid.agents import GridAgentInterface
from contextlib import contextmanager
import warnings


class RLAgentBase(ABC):

    @abstractmethod
    def action_step(self, obs):
        pass

    @abstractmethod
    def save_step(self, obs, act, rew, done):
        pass

    @abstractmethod
    def start_episode(self):
        pass

    @abstractmethod
    def end_episode(self):
        pass

    @contextmanager
    def episode(self):
        self.start_episode()
        yield self
        self.end_episode()

class Agent(RLAgentBase):
    save_modules = []
    def __init__(self, observation_space, action_space, metadata={}, train_history=[], logger=None):

        self.logger = logger
        self.observation_space = self.ensure_space(observation_space)
        self.action_space = self.ensure_space(action_space)
        self.learning_config = {}
        self.model_config = {}
        self.metadata = metadata
        self.train_history = train_history
        self.updates_since_history_update = 0

    @property
    def _save_state(self):
        return {
            'class': self.__class__.__name__,
            'observation_space': space_to_dict(self.observation_space),
            'action_space': space_to_dict(self.action_space),
            'learning_config': self.learning_config,
            'model_config': self.model_config,
            'metadata': self.metadata,
            'train_history': self.updated_train_history,
        }

    @property
    def updated_train_history(self):
        return [*self.train_history, 
        {
            'metadata': self.metadata,
            'learning_config': self.learning_config,
        }]

    def track_gradients(self, module, log_frequency=10):
        # Weights and biases v. 0.8.32? was crashing when Kamal tried to log gradients using
        # the built-in pytorch hooks and `wandb.watch`. This does a similar thing in a similar
        # way but without crashing.
        self._grad_stats = getattr(self, '_grad_stats', {})
        self._grad_counts = getattr(self, '_grad_counts', {})
        self._grads_updated = False
        
        def monitor_gradient_hook(log_name):
            def log_gradient(grad):
                p_ = lambda x: x.detach().cpu().item()
                update_count = self._grad_counts.get(log_name, 0)
                if update_count%log_frequency==0:
                    self._grad_stats = {
                        **self._grad_stats,
                        f'{log_name}_mean': p_(grad.mean()),
                        f'{log_name}_min': p_(grad.min()),
                        f'{log_name}_max': p_(grad.max()),
                        f'{log_name}_std': p_(grad.std()),
                    }
                    self._grads_updated = True
                self._grad_counts[log_name] = update_count + 1
            return log_gradient


        for k, (name, w) in enumerate(module.named_parameters()):
            w.register_hook(monitor_gradient_hook(name))

    def grad_log_sync(self):
        if hasattr(self, '_grad_stats'):
            self.log('gradient_info', self._grad_stats)
            self._grad_stats = {}
            self._grads_updated = False

    def set_logger(self, logger):
        self.logger = logger

    @staticmethod
    def ensure_space(dict_or_space):
        if isinstance(dict_or_space, dict):
            return dict_to_space(dict_or_space)
        else:
            return dict_or_space

    @abstractmethod
    def set_device(self, dev):
        pass

    @abstractmethod
    def action_step(self, obs):
        pass

    @abstractmethod
    def save_step(self, obs, act, rew, done):
        pass

    @abstractmethod
    def start_episode(self):
        pass

    @abstractmethod
    def end_episode(self):
        pass

    @contextmanager
    def episode(self):
        self.start_episode()
        yield self
        self.end_episode()


    def save(self, save_dir, force=False):
        save_dir = os.path.abspath(os.path.expanduser(save_dir))
        model_path = os.path.join(save_dir, 'model.tar')
        metadata_path = os.path.join(save_dir, 'metadata.json')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for f in (model_path, metadata_path):
            if force is False and os.path.isfile(f):
                raise ValueError(f"Error saving {self.__class__.__name__}: save file \"{f}\" already exists.")

        for f in (model_path, metadata_path):
            if os.path.isfile(f):
                os.remove(f)

        print("Saving modules ", self.save_modules)
        torch.save({mod: getattr(self, mod) for mod in self.save_modules
                    }, model_path)

        # Update the training history before saving metadata.

        json.dump(self._save_state, open(metadata_path, "w"))

    @classmethod
    def load(cls, save_dir):
        print(f"Loading", cls.__name__)
        save_dir = os.path.abspath(os.path.expanduser(save_dir))
        model_path = os.path.join(save_dir, 'model.tar')
        metadata_path = os.path.join(save_dir, 'metadata.json')

        metadata = json.load(open(metadata_path,'r'))

        if metadata['class'] != cls.__name__:
            warnings.warn(f"Attempting to load a {cls.__name__} from a {metadata['class']} checkpoint.")
        ret = cls(**{k:v for k,v in metadata.items() if k != 'class'})

        modules_dict = torch.load(model_path)
        
        for k,v in modules_dict.items():
            try:
                getattr(ret, k).load_state_dict(v.state_dict())
            except:
                rsd = getattr(ret, k).state_dict()
                for net_name, net_params in v.state_dict().items():
                    print(net_name, tuple(net_params.shape), tuple(rsd[net_name].shape))
                print(f"Error loading {k}")
                import pdb; pdb.set_trace()
        del modules_dict
        return ret


class IndependentAgents(RLAgentBase):
    def __init__(self, *agents):
        self.agents = list(agents)
        self.observation_space = combine_spaces(
            [agent.observation_space for agent in agents]
        )
        self.action_space = combine_spaces(
            [agent.action_space for agent in agents]
        )
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger
        if logger is not None:
            for k, agent in enumerate(self.agents):
                agent.set_logger(logger.sub_logger(f'agent_{k}'))

    def action_step(self, obs_array):
        return [agent.action_step(obs) for agent, obs in zip(self.agents, obs_array)]

    def set_device(self, dev):
        for agent in self.agents:
            agent.set_device(dev)

    def save_step(self, obs, act, rew, done):
        
        if np.isscalar(done):
            done = np.full(rew.shape, done, dtype='bool')
        elif np.prod(rew.shape)/np.prod(done.shape) == len(self.agents):
            done = (done * np.ones((len(self.agents),1))).astype(done.dtype)
        else:
            done.reshape(rew.shape)

        for k, agent in enumerate(self.agents):
            agent.save_step(obs[k], act[k], rew[k], done[k])

    def start_episode(self, *args, **kwargs):
        for agent in self.agents:
            agent.start_episode(*args, **kwargs)

    def end_episode(self, *args, **kwargs):
        for agent in self.agents:
            agent.end_episode(*args, **kwargs)

    # @property
    # def active(self):
    #     return np.array([agent.active for agent in self.agents], dtype=np.bool)

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, key):
        return self.agents[key]

    def __iter__(self):
        return self.agents.__iter__()

    @contextmanager
    def episode(self):
        self.start_episode()
        yield self
        self.end_episode()
        
    def save(self, path, force=False):
        path = os.path.abspath(os.path.expanduser(path))
        metadata_file = os.path.join(path, 'multi_agent_meta.json')
        if not os.path.isdir(path):
            os.makedirs(path)
        if force and os.path.isfile(metadata_file):
            os.remove(metadata_file)
        json.dump({
            'n_agents': len(self.agents)
            }, fp = open(metadata_file,'w')
        )
        
        keys = [f'{x}' for x in range(len(self.agents))]

        for agent, key in zip(self.agents, keys):
            print(os.path.join(path, key))
            agent.save(os.path.join(path, key), force=force)

    @classmethod
    def load(cls, path, agent_class):
        path = os.path.abspath(os.path.expanduser(path))
        metadata_file = os.path.join(path, 'multi_agent_meta.json')
        metadata = json.load(fp=open(metadata_file,'r'))
        n_agents = int(metadata['n_agents'])
        keys = [f'{x}' for x in range(n_agents)]

        if not isinstance(agent_class, list):
            agent_classes = [agent_class for _ in range(len(keys))]
        else:
            agent_classes = agent_class

        agents = [
            agent_class.load(os.path.join(path, key))
            for agent_class, key in zip(agent_classes, keys)
        ]

        return cls(*agents)

