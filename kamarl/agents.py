import os, sys
import json
import torch
import numpy as np
import types
import gym
from abc import ABC, abstractmethod, abstractproperty
from contextlib import contextmanager
import copy
import warnings
from io import BytesIO, TextIOWrapper
import tempfile
from collections import defaultdict

from urllib.parse import urlparse

import boto3

from marlgrid.agents import GridAgentInterface
import kamarl
from kamarl.utils import (
    space_to_dict,
    dict_to_space,
    combine_spaces,
    update_config
)
# from kamarl.ppo_rec import PPOAEAgent
# from kamarl.ppo import PPOAgent

def parse_s3_uri( s3_uri):
    o = urlparse(s3_uri,  allow_fragments=False)
    return {'bucket':o.netloc, 'key':o.path.strip('/')}

def get_file_from_s3( s3_path):
    s3info = parse_s3_uri(s3_path)
    s3_bucket, s3_key = s3info['bucket'], s3info['key']
    with tempfile.NamedTemporaryFile(delete=False) as f:
        boto3.client('s3').download_fileobj(s3info['bucket'], s3info['key'], f)
    return f.name

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
    def __init__(self, observation_space, action_space, metadata=None, train_history=None, counts=None, logger=None):
        if metadata is None: metadata = {}
        if train_history is None: train_history = []
        if counts is None: counts = defaultdict(int)

        self.logger = logger
        self.observation_space = self.ensure_space(observation_space)
        self.action_space = self.ensure_space(action_space)
        self.learning_config = {}
        self.model_config = {}
        self.metadata = metadata
        self.train_history = train_history
        self.updates_since_history_update = 0
        self.counts = defaultdict(int)

        self.should_log_gradients = False
        self._grad_stats = {}

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
            'counts': dict(self.counts)
        }

    def reset_counts(self):
        self.counts = defaultdict(int)

    @property
    def updated_train_history(self):
        return [*self.train_history, 
        {
            'metadata': self.metadata,
            'learning_config': self.learning_config,
        }]

    def track_gradients(self, module):
        # Weights and biases v. 0.8.32? was crashing when Kamal tried to log gradients using
        # the built-in pytorch hooks and `wandb.watch`. This does a similar thing in a similar
        # way but without crashing.
        def monitor_gradient_hook(log_name):
            def log_gradient(grad):
                p_ = lambda x: x.detach().cpu().item()
                if self.should_log_gradients:
                    self._grad_stats = {
                        **self._grad_stats,
                        f'{log_name}_mean': p_(grad.mean()),
                        f'{log_name}_l1': p_(grad.abs().mean()),
                        f'{log_name}_l2': p_(((grad**2).mean())**0.5),
                        f'{log_name}_min': p_(grad.min()),
                        f'{log_name}_max': p_(grad.max()),
                        f'{log_name}_std': p_(grad.std()),
                    }
            return log_gradient

        for k, (name, w) in enumerate(module.named_parameters()):
            w.register_hook(monitor_gradient_hook(name))

    # def grad_log_sync(self):
    #     if hasattr(self, '_grad_stats'):
    #         self.log('gradients', self._grad_stats)
    #         self._grad_stats = {}
    #         self._grads_updated = False

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
        print("Saving checkpoint:", self.save_modules)
        if save_dir.lower().startswith('s3'):
            print(f"  (S3) --> {save_dir}")
            self.save_s3(save_dir, force=force)
        else:
            print(f"  (local) --> {save_dir}")
            self.save_disk(save_dir, force=force)
        
    def save_s3(self, save_dir, force=False):
        import boto3

        model_data = BytesIO()
        torch.save({mod: getattr(self, mod) for mod in self.save_modules}, model_data)
        model_data.seek(0)
        model_place = parse_s3_uri(os.path.join(save_dir, 'model.tar'))
        boto3.client('s3').put_object(Bucket=model_place['bucket'], Key=model_place['key'], Body=model_data)

        meta_place = parse_s3_uri(os.path.join(save_dir, 'metadata.json'))
        boto3.client('s3').put_object(Bucket=meta_place['bucket'], Key=meta_place['key'], Body=str(json.dumps(self._save_state)))

    def save_disk(self, save_dir, force=False):
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

        torch.save({mod: getattr(self, mod) for mod in self.save_modules
                    }, model_path)

        # Update the training history before saving metadata.

        json.dump(self._save_state, open(metadata_path, "w"))

    @classmethod
    def get_class_to_load(cls, metadata):
        print(f"LAODING {cls}")
        return cls
        # return kamarl.PPOAEAgent
        # if metadata['class'] != cls.__name__:
        #     warning_text = f"Attempting to load a {cls.__name__} from a {metadata['class']} checkpoint."
        #     if metadata['class']=='PPOAgent':
        #         target_class = kamarl.PPOAgent
        #     elif metadata['class']=='PPOAEAgent':
        #         target_class = kamarl.PPOAEAgent
        #     else:
        #         err_msg = f"Not sure how to load an agent with class {metadata['class']}"
        #         raise ValueError(err_msg)
        #     warning_text += f" --> Using {target_class.__name__} instead."
        #     warnings.warn(warning_text)
        # else:
        #     target_class = cls
        # return target_class

    @staticmethod
    def partial_load_state_dict(mod_state, state_dict):
        # mod_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in mod_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            elif isinstance(param, dict):
                if name not in mod_state:
                    mod_state[name] = copy.deepcopy(param)
                else:
                    mod_state[name] = Agent.partial_load_state_dict(mod_state.get(name, {}), param)
            else:
                try:
                    mod_state[name].copy_(param)
                except:
                    raise ValueError(f"Couldn't load param {name} \n>from state\n {mod_state[name]}\n >to\n {param}")

    @classmethod
    def _load(cls, metadata, model_path, device=None):
        target_class = cls.get_class_to_load(metadata)
        ret = target_class(**{k:v for k,v in metadata.items() if k != 'class'})

        if device is None:
            device = getattr(ret, 'device', None)

        modules_dict = torch.load(model_path, map_location=device)

        for k,v in modules_dict.items():
            if k == 'optimizer' and metadata['class'] != cls.__name__:
                print(f"Changed model class from {metadata['class']} to {cls.__name__}. Gonna not load optimizer.")
                continue
            try:
                getattr(ret, k).load_state_dict(v.state_dict())
            except:
                try:
                    cls.partial_load_state_dict(getattr(ret, k).state_dict(), v.state_dict())
                except:
                    if k == 'optimizer':
                        warnings.warn("Failed to load optimizer.")
                    continue
        del modules_dict
        return ret

    @classmethod
    def load_s3(cls, save_path, config_changes = None, device=None):
        if config_changes is None:
            config_changes = {}
        model_path = get_file_from_s3(os.path.join(save_path, 'model.tar'))
        metadata = update_config(
            json.load(open(get_file_from_s3(os.path.join(save_path, 'metadata.json')),'r')),
            config_changes
        )
        return cls._load(metadata, model_path, device=device)


    @classmethod
    def load(cls, save_dir, config_changes = None, device=None):
        if config_changes is None:
            config_changes = {}
        print(f"Loading", cls.__name__)
        save_dir = os.path.abspath(os.path.expanduser(save_dir))
        model_path = os.path.join(save_dir, 'model.tar')
        metadata_path = os.path.join(save_dir, 'metadata.json')

        metadata = update_config(json.load(open(metadata_path,'r')), config_changes)

        return self._load(metadata, model_path, device=device)


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
        # print(done)
        done = np.array(done)
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

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, key):
        return self.agents[key]

    def __iter__(self):
        return self.agents.__iter__()

    def replace_agent(self, agent_no, new_agent):
        self.agents[agent_no] = None
        self.agents[agent_no] = new_agent
        new_agent.set_logger(logger.sub_logger(f'agent_{agent_no}'))


    @contextmanager
    def episode(self):
        self.start_episode()
        yield self
        self.end_episode()
        
    def save(self, path, force=False):
        if not path.lower().startswith('s3'):
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

