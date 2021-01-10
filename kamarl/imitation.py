import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle
import warnings

from collections import defaultdict

import gym

from kamarl.modules import ConvNet, SeqLSTM, make_mlp, device_of, PixelDropout
from kamarl.buffers import RecurrentReplayMemory, init_array_recursive
from kamarl.agents import Agent
from kamarl.utils import space_to_dict, dict_to_space, get_module_inputs, chunked_iterable, discount_rewards, discount_rewards_tensor, update_config_dict

import copy

from .ppo_rec import PPOLSTM, parallel_repeat

class BCAgent(Agent):
    default_learning_config = {
            "max_episode_length": 1000,
            "hidden_update_interval": 5, # minibatches/gradient steps
            "hidden_update_n_parallel": 32,
            "batch_size": 1e5, # steps
            "updates_per_batch": 1e2,
            "minibatch_size": 256,
            "minibatch_seq_len": 10,
            'learning_rate': 3.e-4
    }
    default_model_config = {
        # The default values for the model configuration are set 
        #  in the PPOLSTM class. Values set here would overwrite
        #  those defaults.
    }
    save_modules = ['policy', 'optimizer']
    def __init__(self, expert_agent=None, learning_config={}, model_config={}, metadata={}, train_history=[]):
        super().__init__(
            observation_space=expert_agent.observation_space,
            action_space=expert_agent.action_space,
            metadata=metadata,
            train_history=train_history)

        self.learning_config, novel_keys = update_config_dict(self.default_learning_config, learning_config)
        if len(novel_keys) > 0:
            warnings.warn(f"Specified unknown keys in {self.__class__.__name__} learning config: {novel_keys}")
        
        self.expert_agent = expert_agent
        if expert_agent is not None:
            self.expert_agent.training = False

        self.policy = PPOLSTM(self.expert_agent.observation_space, self.expert_agent.action_space, config=model_config)
        self.model_config = self.policy.config
        
        # self.track_gradients(self.policy)
        self.training = True

        self.replay_memory_array_specs = {
                'obs': self.observation_space,
                'act': self.action_space,
                'rew': ((), "float32"),
                'done': ((), "bool"),
                "hx_cx": ((2, self.policy.lstm.hidden_size), "float32")
            }

        self.replay_memory = RecurrentReplayMemory(
            self.replay_memory_array_specs,
            max_episode_length=self.learning_config["max_episode_length"]+1,
            max_num_steps=self.learning_config["batch_size"] + self.learning_config["max_episode_length"]+1,
        )
        self.device = torch.device('cpu')

        self.n_parallel = None

        self.reset_optimizer()
        self.reset_hidden()
        self.reset_state()
        self.reset_info()
        self.counts = defaultdict(int)

    @property
    def updated_train_history(self):
        return [*self.train_history, 
        {
            'metadata': self.metadata,
            'learning_config': self.learning_config,
            'n_updates': self.counts['updates'],
            'n_episodes': self.counts['episodes']
        }]
    @property
    def config(self):
        return {
            'observation_space': space_to_dict(self.observation_space),
            'action_space': space_to_dict(self.action_space),
            'learning_config': self.learning_config,
            'model_config': self.policy.config,
        }

    def reset_info(self):
        if self.expert_agent is not None:
            self.expert_agent.reset_info()
        self.logged_lengths = []
        self.logged_rewards = []
        self.logged_streaks = []
        self.logged_reward_counts = []

    def reset_state(self):
        if self.expert_agent is not None:
            self.expert_agent.reset_state()
        self.state = {
            'hx_cx': self.policy.empty_hidden(numpy=True),
        }
        self.active = np.zeros(self.n_parallel, dtype=np.bool)
        self.state = {
            k:parallel_repeat(v, self.n_parallel)
            for k, v in self.state.items()
        }

    def reset_hidden(self):
        if self.expert_agent is not None:
            self.expert_agent.reset_hidden()
        self.hx_cx = parallel_repeat(
            self.policy.empty_hidden(), self.n_parallel
            ).to(self.device)

    def log(self, *args, **kwargs):
        if getattr(self, 'logger', None) is None:
            pass
        else:
            self.logger.log_value(*args, **{**kwargs, 'commit':True})

    def set_device(self, dev):
        if self.expert_agent is not None:
            self.expert_agent.set_device(dev)
        if torch.device(dev) == self.device: 
            return
        else:
            self.device = torch.device(dev)
            self.hx_cx.to(self.device)
            self.policy.to(self.device)
            tmp = self.optimizer.state_dict()
            # self.optimizer.to(self.device)
            self.reset_optimizer()
            self.optimizer.load_state_dict(tmp)

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr = self.learning_config['learning_rate'])

    def action_step(self, X):
        if self.expert_agent is not None:
            return self.expert_agent.action_step(X)

        was_training = self.policy.training
        self.policy.eval()
        last_hx_cx = self.hx_cx.detach().cpu().numpy()
        if self.n_parallel is not None:
            if isinstance(X, dict):
                X = {k:v[:,None,...] for k,v in X.items()}
            else:
                X = X[:,None,...]
            tmp = self.policy.step(X, self.hx_cx.unbind(-2))
            a, v, logp, hx_cx = [x.squeeze() for x in tmp]
            self.hx_cx = torch.stack(hx_cx.unbind(-2))
        else:
            a, v, logp, self.hx_cx = self.policy.step(X, self.hx_cx)

        self.counts['steps'] += 1
        self.counts['episode_steps'] += 1

        if was_training:
            self.policy.train()
        return a

    def save_step(self, obs, act, rew, done, ignore_step=False):
        """ 
        Save an environment transition.
        """
        
        def decollate(val, ix):
            if isinstance(val, dict):
                return {k: decollate(v, ix) for k,v in val.items()}
            else:
                return val[ix]

        if self.training:
            save_dict = {
                'obs': obs, 'act': act, 'rew': rew, 'done': done,
            }
            for i, ep in enumerate(self.active_episodes):
                if not ep['done',-1]:
                    if self.n_parallel is None:
                        ep.append(save_dict)
                    else:
                        ep.append({k: decollate(v,i) for k,v in save_dict.items()})

    def refresh_stale(self, episodes, parallel=32, refresh_hidden=True, return_mse=False):
        torch_mode = episodes[0].tensor_mode
        was_eval = not self.policy.training
        self.policy.train()
        with torch.no_grad():
            update_sum_err = 0
            update_value_count = 0
            for episodes in chunked_iterable(sorted(episodes, key=lambda e: len(e)), size=parallel):
                max_len = max(len(e) for e in episodes)
                data, keys = init_array_recursive(self.replay_memory_array_specs['obs'], (len(episodes), max_len),
                                    array_hook=torch.zeros,  array_kwargs = {'device':self.device})
                
                for i,ep in enumerate(episodes):
                    for key_path in keys:
                        tgt = data
                        src = ep['obs']
                        if len(key_path) > 0:
                            for k in key_path[:-1]:
                                tgt = tgt[k]
                                src = src[k]
                            k = key_path[-1]
                            if isinstance(src[k], np.ndarray):
                                tgt[k][i,:len(src[k]),...] = torch.from_numpy(src[k])
                            else:
                                tgt[k][i,:len(src[k]),...] = src[k]
                        else:
                            if isinstance(src, np.ndarray):
                                tgt[i,:len(src),...] = torch.from_numpy(src)
                            else:
                                tgt[i,:len(src),...] = src


                # Transpose below changes from (hx/cx, ep/batch, seq, hid)
                # to (ep/batch, seq, hx/cx, hid)
                _, new_values, new_hiddens = self.policy.pi_v(data, hx=None, return_hidden=True)
                new_hiddens = new_hiddens.permute(1, 2, 0, 3)

                if not torch_mode:
                    if refresh_hidden and isinstance(ep['hx_cx'], np.ndarray):
                        new_hiddens = new_hiddens.cpu().numpy()
                for i, ep in enumerate(episodes):
                    if return_mse:
                        update_sum_err += ((ep['hx_cx'][1:len(ep)]-new_hiddens[i, :len(ep)-1, :, :])**2).sum()
                        update_value_count += len(ep)-1
                    if refresh_hidden:
                        ep['hx_cx'][1:len(ep)] = new_hiddens[i, :len(ep)-1, :, :]

            if was_eval:
                self.policy.eval()

            if return_mse:
                return update_sum_err/update_value_count

    def compute_loss(self, data):
        mask = data['done'].cumsum(1).cumsum(1)<=1
        N = mask.sum()

        # policy loss
        pi = self.policy.pi(data['obs'], data['hx_cx'])

        logp = (pi.log_prob(data['act'])).mean()

        loss = -logp
        loss_info = {
            'logp': logp,
        }

        return loss, loss_info

    def sample_replay_buffer(self, batch_size, seq_len):
        data = self.replay_memory.sample_sequence(batch_size=batch_size, seq_len=seq_len)
        # print(f"DATA DEVICE IS {dat")
        # import pdb; pdb.set_trace()
        if self.replay_memory.episodes[0].tensor_mode:
            data['hx_cx'] = data['hx_cx'][:,0,...].transpose(-2,0)
            return data
        else:
            # ignore hiddens after the first before sending to GPU.
            data['hx_cx'] = np.moveaxis(data['hx_cx'][:,0,...], -2, 0)
            def to_tensor(x):
                if isinstance(x, dict):
                    return {k: to_tensor(v) for k,v in x.items()}
                return torch.from_numpy(x).to(self.device)
            return to_tensor(data)
                

    def optimize(self):
        self.reset_info()

        if bool(self.training):
            hp = self.learning_config
            device = self.device

            n_hidden_updates = 0

            self.policy.train()

            # for episode in self.replay_memory.episodes:
            #     episode.to_tensor(device=device)

            n_minibatches = 0
            self.refresh_stale(self.replay_memory.episodes, parallel=hp['hidden_update_n_parallel'])
            for i in range(int(hp['updates_per_batch'])):
                # If it's time, recompute the advantages and hidden states in the replay buffer to make sure
                # they don't get too stale.
                hid_up_in = hp["hidden_update_interval"]
                if (hid_up_in is not None) and (i%hid_up_in == 0) and (i > 0):
                    n_hidden_updates += 1
                    self.refresh_stale(self.replay_memory.episodes, parallel=hp['hidden_update_n_parallel'])
                

                # Sample a minibatch of data with which to compute losses/update parameters.
                minibatch_data = self.sample_replay_buffer(batch_size=hp["minibatch_size"], seq_len=hp["minibatch_seq_len"])

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()

                    policy_loss, loss_metrics = self.compute_loss(minibatch_data)

                    policy_loss.backward()

                    self.optimizer.step()

                # Save metrics.
                # pi_infos.append(loss_metrics)
                # policy_losses.append(policy_loss.detach().cpu().numpy())

                mean = lambda vals: np.nan if len(vals) == 0 else np.nanmean(vals) if not isinstance(vals[0], torch.Tensor) else torch.tensor(vals).mean().cpu().item()

                steps = np.array([len(e) for e in self.replay_memory.episodes])

                if self.counts['updates']%10 == 0:
                    log_data = {
                        'update_no': self.counts['updates'],
                        'policy_loss': policy_loss,
                        'logp': loss_metrics['logp'],
                    }
                    self.log('update_data', log_data)
                    print(f"Update {1+self.counts['updates']}: {n_hidden_updates} hidden updates, loss={policy_loss}")

                self.counts['updates'] += 1

        self.replay_memory.clear()
        

    def clear_memory(self):
        # self.replay_memory.clear()
        # self.last_update_episode = self.counts['episodes']
        pass

    def start_episode(self, n_parallel=None):
        if self.expert_agent is not None:
            self.expert_agent.start_episode(n_parallel=n_parallel)
        self.n_parallel = n_parallel
        self.reset_hidden()
        self.reset_state()

        # self.reset_info()
        self.last_val = None

        if self.training:
            self.active_episodes = [
                self.replay_memory.get_new_episode() for x in 
                range(1 if self.n_parallel is None else self.n_parallel)
            ]


        self.was_active = True
        self.counts['episode_steps'] = 0

    def end_episode(self, log=False):
        if self.expert_agent is not None:
            self.expert_agent.end_episode(log=False)

        for ep in self.active_episodes:
            ep.freeze()
            self.replay_memory.add_episode(ep)
            self.counts['episodes'] += 1

        if len(self.replay_memory) >= self.learning_config['batch_size']:
            self.optimize()
            self.last_update_episode = self.counts['episodes']

        # self.grad_log_sync()