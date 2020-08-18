import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import pickle
import warnings

from collections import defaultdict

import gym

from kamarl.modules import ConvNet, DeconvNet, SeqLSTM, make_mlp, device_of
from kamarl.buffers import RecurrentReplayMemory, init_array_recursive
from kamarl.agents import Agent
from kamarl.utils import space_to_dict, dict_to_space, get_module_inputs, chunked_iterable, discount_rewards, discount_rewards_tensor

from PIL import Image
import copy


def update_config_dict(base_config, new_config):
    novel_keys = []
    updated_config = copy.deepcopy(base_config)
    for k,v in new_config.items():
        if k not in base_config:
            novel_keys.append(k)
        updated_config[k] = v
    return updated_config, novel_keys
        


class PPOLSTM(nn.Module):
    default_config = {
        "conv_layers" : [
            {'out_channels': 8, 'kernel_size': 3, 'stride': 3, 'padding': 0},
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        ],
        'input_trunk_layers': [192,192],
        'lstm_hidden_size': 256,
        'val_mlp_layers': [64,64],
        'pi_mlp_layers': [64,64],
        'fancy_init': True
    }
    

    def __init__(
            self,
            observation_space,
            action_space,
            config = {}
        ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"{self.__class__.__name__} only supports discrete action spaces"
            )

        self.config, novel_keys = update_config_dict(self.default_config, config)
        if len(novel_keys) > 0:
            warnings.warn(f"Specified unknown keys in {self.__class__.__name__} model config: ", novel_keys)

        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        input_image_shape, n_flat_inputs = get_module_inputs(self.observation_space)
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.input_keys = [x for x in self.observation_space.spaces.keys() if x != 'pov']
        else:
            self.input_keys = []

        trunk = []

        in_channels = 3 # todo: infer number of image channels from observation space shape.
        conv_layers = []
        for k,c in enumerate(self.config['conv_layers']):
            conv_layers.append(nn.Conv2d(in_channels, **c))
            in_channels = c['out_channels']
            if k < len(self.config['conv_layers']) - 1:
                conv_layers.append(nn.ReLU(inplace=True))

        self.conv_layers = ConvNet(
            *conv_layers,
            input_shape=input_image_shape,
            output_nonlinearity=nn.Tanh,
        )

        self.combined_input_layers = make_mlp(
            [self.conv_layers.n + n_flat_inputs, *self.config['input_trunk_layers']],
            nonlinearity=nn.ReLU,
            output_nonlinearity=nn.Tanh)
        
        tmp = [x.out_features for x in self.combined_input_layers if hasattr(x, 'out_features')]
        if len(tmp) == 0:
            feature_count = self.conv_layers.n+n_flat_inputs
        else:
            feature_count = tmp[-1]

        self.lstm = SeqLSTM(feature_count, self.config['lstm_hidden_size'])
        

        if self.config.get('deconv_layers', None) is not None:
            deconv_layers_config = self.config['deconv_layers']
        else:
            deconv_layers_config = []
            for conv_layer in self.config['conv_layers'][::-1]:
                conv_layer_copy = {k:v for k,v in conv_layer.items()}
                conv_layer_copy['in_channels'] = conv_layer_copy['out_channels']
                del conv_layer_copy['out_channels']
                deconv_layers_config.append(conv_layer_copy)

        deconv_layers = []
        out_channels = 3
        for k,c in enumerate(deconv_layers_config[::-1]):
            deconv_layers.append(nn.ConvTranspose2d(out_channels=out_channels, **c))
            out_channels = c['in_channels']
            if k < len(deconv_layers_config) - 1:
                deconv_layers.append(nn.ReLU(inplace=True))

        self.deconv_layers = DeconvNet(
            *deconv_layers[::-1],
            n_latent = self.config['lstm_hidden_size'] + action_space.n,
            image_size=input_image_shape
        )
        self.mlp_val = make_mlp(
            layer_sizes=[self.lstm.hidden_size, *self.config['val_mlp_layers'], 1],
            nonlinearity=nn.Tanh,
            output_nonlinearity=None
        )

        self.mlp_pi = make_mlp(
            layer_sizes=[self.lstm.hidden_size, *self.config['pi_mlp_layers'], action_space.n],
            nonlinearity=nn.Tanh,
            output_nonlinearity=None
        )

        if self.config['fancy_init']:
            for mod in self.modules():
                if hasattr(mod, '_init_parameters'):
                    mod._init_parameters()

    def empty_hidden(self, numpy=False):
        if numpy:
            return np.zeros((2, self.config['lstm_hidden_size']), dtype=np.float32)
        else:
            return torch.zeros(
                (2, self.config['lstm_hidden_size']),
                dtype=torch.float32,
                device=device_of(self))

    def process_input(self, X):
        if isinstance(self.observation_space, gym.spaces.Box):
            return (X,torch.tensor((), device=device_of(self)))
        elif isinstance(self.observation_space, gym.spaces.Dict):
            batch_dims = X['pov'].shape[:-3] # dumb hack.
            def expand_dims(X):
                while len(X.shape) < len(batch_dims) + 1:
                    X = X[..., None]
                return X
            if len(self.input_keys)>0:
                try:
                    return (
                        X['pov'], 
                        torch.cat(tuple([
                            (
                                F.one_hot(torch.tensor(X[k]), self.observation_space[k].n).float() 
                                    if isinstance(self.observation_space[k], gym.spaces.Discrete)
                                else expand_dims(torch.tensor(X[k])).float()
                            )
                            for k in self.input_keys
                        ]), dim=-1).to(device_of(self))
                    )
                except:
                    tmp = [
                            (
                                F.one_hot(torch.tensor(X[k]), self.observation_space[k].n).float() 
                                    if isinstance(self.observation_space[k], gym.spaces.Discrete)
                                else expand_dims(torch.tensor(X[k])).float()
                            )
                            for k in self.input_keys
                        ]
                    import pdb; pdb.set_trace()
            else:
                return (X['pov'],None)
        else:
            raise ValueError(f"Can't process input of type {type(X)}")

    def input_layers(self, X):
        if isinstance(X, dict) and 'obs' in X:
            X = X['obs']
        X_image, X_other = self.process_input(X)
        X = self.conv_layers(X_image)
        if X_other is not None:
            X = torch.cat([X, X_other], dim=-1)
        return self.combined_input_layers(X)

    def compute_hidden(self, X):
        return self.lstm(self.input_layers(X))
        
    def pi_v(self, X, hx, return_hidden=False):
        X = self.input_layers(X)
        hx_cx_new = self.lstm(X, hx, vec_hidden=False)

        pi = torch.distributions.Categorical(logits=self.mlp_pi(hx_cx_new[0]))
        v = self.mlp_val(hx_cx_new[0])
        if return_hidden:
            return pi, v, hx_cx_new
        else:
            return pi, v

    def pi_v_rec(self, X, hx, act, return_hidden=False, input_dropout=0.0):
        X = self.input_layers(X)
        X = nn.functional.dropout(X, p=input_dropout)

        hx_cx_new = self.lstm(X, hx, vec_hidden=False)

        pi = torch.distributions.Categorical(logits=self.mlp_pi(hx_cx_new[0]))
        v = self.mlp_val(hx_cx_new[0])
        rec = self.deconv_layers(torch.cat((hx_cx_new[0],F.one_hot(act, self.action_space.n).float()),dim=-1))
        return pi, v, rec

    def step(self, X, hx=None):
        X = self.input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)

        policy_logits = self.mlp_pi(X)
        pi = torch.distributions.Categorical(logits=policy_logits)
        act = pi.sample()
        logp = pi.log_prob(act)
        val = self.mlp_val(X)

        return act.cpu().numpy(), val, logp, torch.stack((X, hx))

def parallel_repeat(value, n_parallel=None):
    if n_parallel is None:
        return value
    if isinstance(value, torch.Tensor):
        return torch.repeat_interleave(value[None,...], n_parallel, dim=0)
    else:
        return np.repeat(np.array(value)[None,...], n_parallel, axis=0)

class PPOAEAgent(Agent):
    default_learning_config = {
            'num_minibatches': 10,
            'min_num_minibatches': 1,
            "max_episode_length": 1000,
            "batch_size": 25, # episodes
            "hidden_update_interval": 5, # minibatches!
            "hidden_update_n_parallel": 32,
            "minibatch_size": 256,
            "minibatch_seq_len": 10,
            'learning_rate': 3.e-4,
            "kl_target": 0.01,
            "kl_hard_limit": 0.03,
            "clamp_ratio": 0.2,
            "lambda":0.97,
            'entropy_bonus_coef': 0.001,
            'policy_loss_coef': 1.0,
            'value_loss_coef': 0.5,
            'reconstruction_loss_coef': 1.0,
            'reconstruction_loss_loss': 'l2',
            "gamma": 0.99,

            "bootstrap_values": True,

            'predict_this_frame': False,

            'save_test_image': None,
            'lstm_train_hidden_dropout': 0.0,
            'lstm_train_input_dropout': 0.0,
    }
    default_model_config = {
        # The default values for the model configuration are set 
        #  in the PPOLSTM class. Values set here would overwrite
        #  those defaults.
    }
    save_modules = ['ac', 'optimizer']
    def __init__(self, observation_space, action_space, learning_config=None, model_config=None, metadata=None, train_history=None, counts=None):
        if learning_config is None: learning_config = {}
        if model_config is None: model_config = {}
        if metadata is None: metadata = {}
        if train_history is None: train_history = []

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            metadata=metadata,
            train_history=train_history,
            counts=counts)

        self.learning_config, novel_keys = update_config_dict(self.default_learning_config, learning_config)
        if len(novel_keys) > 0:
            warnings.warn(f"Specified unknown keys in {self.__class__.__name__} learning config: ", novel_keys)
        
        self.ac = PPOLSTM(self.observation_space, self.action_space, config=model_config)
        # self.ac_backup = copy.deepcopy(self.ac)

        self.model_config = self.ac.config
        
        # self.track_gradients(self.ac)
        self.training = True

        self.replay_memory_array_specs = {
                'obs': self.observation_space,
                'act': self.action_space,
                'rew': ((), "float32"),
                'done': ((), "bool"),
                "val": ((), "float32"),
                "adv": ((), "float32"),
                "ret": ((), "float32"),
                "logp": ((), "float32"),
                "hx_cx": ((2, self.ac.lstm.hidden_size), "float32")
            }

        self.replay_memory = RecurrentReplayMemory(
            self.replay_memory_array_specs,
            max_episode_length=self.learning_config["max_episode_length"]+1,
            max_num_steps=(self.learning_config["max_episode_length"]+1) * (self.learning_config['batch_size']+1)
        )
        self.device = torch.device('cpu')

        self.n_parallel = None

        self.reset_optimizer()
        self.reset_hidden()
        self.reset_state()
        self.reset_info()

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
            'model_config': self.ac.config,
        }

    def reset_info(self):
        self.logged_lengths = []
        self.logged_rewards = []
        self.logged_streaks = []
        self.logged_reward_counts = []

    def reset_state(self):
        self.state = {
            'hx_cx': self.ac.empty_hidden(numpy=True),
            'val': 0,
            'adv': 0,
            'ret': 0,
            'logp': 0
        }
        self.active = np.zeros(self.n_parallel, dtype=np.bool)
        self.state = {
            k:parallel_repeat(v, self.n_parallel)
            for k, v in self.state.items()
        }

    def reset_hidden(self):
        self.hx_cx = parallel_repeat(
            self.ac.empty_hidden(), self.n_parallel
            ).to(self.device)

    def log(self, *args, **kwargs):
        if getattr(self, 'logger', None) is None:
            pass
        else:
            self.logger.log_value(*args, **{**kwargs, 'commit':True})

    def set_device(self, dev):
        if torch.device(dev) == self.device: 
            return
        else:
            self.device = torch.device(dev)
            self.hx_cx.to(self.device)
            self.ac.to(self.device)
            tmp = self.optimizer.state_dict()
            # self.optimizer.to(self.device)
            self.reset_optimizer()
            self.optimizer.load_state_dict(tmp)

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr = self.learning_config['learning_rate'])

    def action_step(self, X):
        self.ac.eval()
        last_hx_cx = self.hx_cx.detach().cpu().numpy()
        if self.n_parallel is not None:
            if isinstance(X, dict):
                X = {k:v[:,None,...] for k,v in X.items()}
            else:
                X = X[:,None,...]
            tmp = self.ac.step(X, self.hx_cx.unbind(-2))
            a, v, logp, hx_cx = [x.squeeze() for x in tmp]
            self.hx_cx = torch.stack(hx_cx.unbind(-2))
        else:
            a, v, logp, self.hx_cx = self.ac.step(X, self.hx_cx)

        self.state = {**self.state,
            'val': v.detach().cpu().numpy(),
            'logp': logp.detach().cpu().numpy(),
            'hx_cx': last_hx_cx
        }

        self.counts['steps'] += 1
        self.counts['episode_steps'] += 1

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

        # Keep track of "streaks" of positive rewards.
        still_going = ~np.array(done).astype(bool)
        rew_ = np.array(rew)*still_going
        min_streak_rew = 1.0
        self.episode_reward_streaks = (self.episode_reward_streaks + (rew_ >= min_streak_rew))*(rew_ >= 0)
        self.logged_reward_counts[-1] += (rew_>=min_streak_rew)
        self.logged_streaks[-1] = np.maximum(self.episode_reward_streaks, self.logged_streaks[-1])
        self.logged_rewards[-1] += rew*still_going
        self.logged_lengths[-1] += still_going

        if self.training:
            save_dict = {
                'obs': obs, 'act': act, 'rew': rew, 'done': done,
                **self.state
            }
            for i, ep in enumerate(self.active_episodes):
                if not ep['done',-1]:
                    if self.n_parallel is None:
                        ep.append(save_dict)
                    else:
                        ep.append({k: decollate(v,i) for k,v in save_dict.items()})

    def refresh_stale(self, episodes, parallel=32, refresh_hidden=True, refresh_adv=True, return_mse=False, torch_mode=False):
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
                _, new_values, new_hiddens = self.ac.pi_v(data, hx=None, return_hidden=True)
                new_hiddens = new_hiddens.permute(1, 2, 0, 3)

                if not torch_mode:
                    if refresh_hidden and isinstance(ep['hx_cx'], np.ndarray):
                        new_hiddens = new_hiddens.cpu().numpy()

                    if refresh_adv and isinstance(ep['val'], np.ndarray):
                        new_values = new_values.cpu().numpy()

                for i, ep in enumerate(episodes):
                    if return_mse:
                        update_sum_err += ((ep['hx_cx'][1:len(ep)]-new_hiddens[i, :len(ep)-1, :, :])**2).sum()
                        update_value_count += len(ep)-1
                    if refresh_hidden:
                        ep['hx_cx'][1:len(ep)] = new_hiddens[i, :len(ep)-1, :, :]
                    if refresh_adv:
                        ep['val'][:] = new_values[i, :len(ep), 0]
                        self.calculate_advantages(ep)
        if refresh_adv:
            self.normalize_advantages()
        if return_mse:
            return update_sum_err/update_value_count

    def calculate_advantages(self, episode, last_val=0):
        ''' Populate advantages and returns in an episode. '''
        hp = self.learning_config

        if hp['bootstrap_values']:
            last_val = episode.val[-1]
        else:
            last_val = 0


        if episode.tensor_mode:
            rew = torch.cat((episode.rew, episode.rew.new_tensor([last_val])))
            vals = torch.cat((episode.val, episode.val.new_tensor([last_val])))
            deltas = rew[:-1] + hp['gamma'] * (vals[1:] - vals[:-1])
            episode['adv',:] = discount_rewards_tensor(deltas,hp['gamma']*hp['lambda'])
            episode['ret',:] = discount_rewards_tensor(rew, hp['gamma'])[:-1]
        else:
            rew = np.append(episode.rew, np.array(last_val, dtype=episode.rew.dtype))
            vals = np.append(episode.val, np.array(last_val, dtype=episode.val.dtype))
            deltas = rew[:-1] + hp['gamma'] * (vals[1:] - vals[:-1])
            episode['adv',:] = discount_rewards(deltas, hp['gamma']*hp['lambda'])
            episode['ret',:] = discount_rewards(rew, hp['gamma'])[:-1]

        return episode

    def normalize_advantages(self):
        if self.replay_memory.episodes[0].tensor_mode:
            advantages = torch.cat(tuple(ep.adv for ep in self.replay_memory.episodes))
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            for ep in self.replay_memory.episodes:
                ep['adv',:] = (ep['adv',:] - adv_mean) / adv_std
        else:
            advantages = np.concatenate([ep.adv for ep in self.replay_memory.episodes])
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            for ep in self.replay_memory.episodes:
                ep['adv',:] = (ep['adv',:] - adv_mean) / adv_std

    def compute_loss(self, data):
        mask = data['done'].cumsum(1).cumsum(1)<=1
        N = mask.sum()
        clamp_ratio = self.learning_config['clamp_ratio']

        # policy loss
        pi, v_theta, rec = self.ac.pi_v_rec(data['obs'], data['hx_cx'], data['act'], 
            input_dropout=self.learning_config['lstm_train_input_dropout'])

        logp = pi.log_prob(data['act'])

        ratio = torch.exp(logp - data['logp'])
        clip_adv = torch.clamp(ratio, 1-clamp_ratio, 1+clamp_ratio) * data['adv']
        loss_pi = -((torch.min(ratio * data['adv'], clip_adv))*mask).sum()/N

        loss_val = (((v_theta.squeeze() - data['ret'])**2)*mask).sum()/N

        approx_kl = (((data['logp'] - logp)*mask).sum()/N)

        loss_ent = - pi.entropy().sum()/N
        loss_val = loss_val

        mask2 = mask[:,:-1]

        if bool(self.learning_config['predict_this_frame']):
            loss_rec = (data['obs']['pov'][:, :-1,...]/255. - rec[:,:-1,...]).view(mask2.shape[0],mask2.shape[1],-1)
        else:
            loss_rec = (data['obs']['pov'][:, 1:,...]/255. - rec[:,:-1,...]).view(mask2.shape[0],mask2.shape[1],-1)

        if (self.learning_config['reconstruction_loss_loss'] is not None) and (self.learning_config['reconstruction_loss_loss'] == 'l1'):
            loss_rec = (((loss_rec.abs()).mean(-1)*mask2).sum())/mask2.sum()
        else:
            loss_rec = (((loss_rec**2.0).mean(-1)*mask2).sum())/mask2.sum()

        ent = pi.entropy().sum().item()/N
        clipped = ratio.gt(1+clamp_ratio) | ratio.lt(1-clamp_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).sum()/N
        pi_info = {
            'kl': approx_kl.detach().cpu().item(),
            'ent': ent.detach().cpu().item(),
            'cf': clipfrac.detach().cpu().item(),
            'sample_prediction': rec[0,0]
        }

        return loss_pi, loss_val, loss_ent, loss_rec, pi_info

    def check_kl(self, ac, data):
        with torch.no_grad():
            mask = data['done'].cumsum(1).cumsum(1)<=1
            N = mask.sum()
            clamp_ratio = self.learning_config['clamp_ratio']

            # policy loss
            pi, v_theta = self.ac.pi_v(data['obs'], data['hx_cx'])

            logp = pi.log_prob(data['act'])

            approx_kl = (((data['logp'] - logp)*mask).sum()/N).item()
        return approx_kl

    def sample_replay_buffer(self, batch_size, seq_len):
        data = self.replay_memory.sample_sequence(batch_size=batch_size, seq_len=seq_len)
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
        log_data = {
            'update_no': self.counts['updates'],
            'ep_no': self.counts['episodes'],
            'step_no': self.counts['steps'],

            'mean_reward_streak': np.mean(self.logged_streaks),
            'max_reward_streak': np.max(self.logged_streaks),
            'reward_streak_15p': np.quantile(self.logged_streaks, 0.15),
            'reward_streak_85p': np.quantile(self.logged_streaks, 0.85),

            'mean_reward': np.mean(self.logged_rewards),
            'ep_return_mean': np.mean(self.logged_rewards),
            'ep_return_15p': np.quantile(self.logged_rewards, 0.15),
            'ep_return_85p': np.quantile(self.logged_rewards, 0.85),
            'ep_return_std': np.std(self.logged_rewards),
            'ep_return_min': np.min(self.logged_rewards),
            'ep_return_max': np.max(self.logged_rewards),

            'mean_length': np.mean(self.logged_lengths),

            'mean_reward_counts': np.mean(self.logged_reward_counts),
            'reward_counts_15p': np.quantile(self.logged_reward_counts, 0.15),
            'reward_counts_85p': np.quantile(self.logged_reward_counts, 0.85),

            'mean_streak_fraction': np.mean(np.array(self.logged_streaks)/np.array(self.logged_reward_counts).clip(1)),
        }
        self.reset_info()

        if bool(self.training):
            opt_start_time = time.time()
            hp = self.learning_config
            device = self.device

            pi_infos = []
            critic_losses = []
            policy_losses = []
            entropy_losses = []
            reconstruction_losses = []

            n_hidden_updates = 0

            final_minibatch_kl = 0
            terminal_kl = 0

            self.normalize_advantages()
            self.ac.train()

            for episode in self.replay_memory.episodes:
                episode.to_tensor(device=device)

            n_minibatches = 0
            for i in range(hp['num_minibatches']):

                # If it's time, recompute the advantages and hidden states in the replay buffer to make sure
                # they don't get too stale.
                hid_up_in = hp["hidden_update_interval"]
                if (hid_up_in is not None) and (i%hid_up_in == 0) and (i > 0):
                    n_hidden_updates += 1
                    self.refresh_stale(self.replay_memory.episodes, parallel=hp['hidden_update_n_parallel'])
                    self.normalize_advantages()

                
                # Back up the network parameters and the optimizer state after the minimum number of minibatches.
                # If the policy later drifts too far, this backup will be used to undo the subsequent gradient 
                # updates that led to the policy diverging.
                if i == hp['min_num_minibatches']:
                    backup_i = i
                    backup_weights = self.ac.state_dict()
                    backup_opt_state = self.optimizer.state_dict()
                    # Note: if min_num_minibatches isn't a multiple of hidden_update_interval,
                    # the kl estimate saved below might be a bit stale.
                    backup_kl_check = self.check_kl(self.ac, self.sample_replay_buffer(batch_size=512, seq_len=8))

                # Sample a minibatch of data with which to compute losses/update parameters.
                minibatch_data = self.sample_replay_buffer(batch_size=hp["minibatch_size"], seq_len=hp["minibatch_seq_len"])

                # Apply dropout to LSTM "hx" (but not "cx")
                nn.functional.dropout(minibatch_data['hx_cx'][0,...], p=hp['lstm_train_hidden_dropout'], inplace=True)

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()

                    policy_loss, critic_loss, entropy_loss, reconstruction_loss, loss_metrics = self.compute_loss(minibatch_data)

                    # If est_kl(current_policy, policy_at_start_of_updates) have diverged, then stop updating parameters.
                    if i>hp['min_num_minibatches'] and loss_metrics['kl'] > 1.5 * hp['kl_target']:
                        break

                    ( policy_loss * self.learning_config['policy_loss_coef'] +
                      entropy_loss * self.learning_config['entropy_bonus_coef'] * self.learning_config['policy_loss_coef'] +
                      critic_loss * self.learning_config['value_loss_coef'] + 
                      reconstruction_loss * self.learning_config['reconstruction_loss_coef']
                    ).backward()

                    self.optimizer.step()

                # This will be the estimated kl from the last minibatch before potential early termination.
                final_minibatch_kl = loss_metrics['kl']

                # Save metrics.
                pi_infos.append({k:v for k,v in loss_metrics.items() if k != 'sample_prediction'})
                critic_losses.append(critic_loss.detach().cpu().numpy())
                policy_losses.append(policy_loss.detach().cpu().numpy())
                entropy_losses.append(entropy_loss.detach().cpu().numpy())
                reconstruction_losses.append(reconstruction_loss.detach().cpu().numpy())

                # Number of minibatch iterations/gradient updates that actually took place.
                n_minibatches += 1

            if hp['save_test_image'] is not None:
                an_image = (loss_metrics['sample_prediction'].detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                print("About an image: ", an_image.mean(), an_image.min(), an_image.max())
                Image.fromarray(an_image).save(hp['save_test_image'])

            # After the minibatch updates are finished, do a final check to make sure the policy hasn't diverged too much.
            # If it has, then reset the actor, critic, and optimizer parameters to the backup values saved above.
            used_backup = False
            if hp['kl_hard_limit'] is not None:
                self.refresh_stale(self.replay_memory.episodes)
                kl_after = self.check_kl(self.ac, self.sample_replay_buffer(batch_size=512, seq_len=8))

                if kl_after > hp['kl_hard_limit']:
                    used_backup = True
                    self.ac.load_state_dict(backup_weights)
                    self.optimizer.load_state_dict(backup_opt_state)
                    
                    kl_after = backup_kl_check
                    n_minibatches = backup_i


            policy_losses = policy_losses[:n_minibatches]
            critic_losses = critic_losses[:n_minibatches]
            entropy_losses = entropy_losses[:n_minibatches]
            reconstruction_losses = reconstruction_losses[:n_minibatches]
            pi_infos = pi_infos[:n_minibatches]


            mean = lambda vals: np.nan if len(vals) == 0 else np.nanmean(vals) if not isinstance(vals[0], torch.Tensor) else torch.tensor(vals).mean().cpu().item()

            steps = np.array([len(e) for e in self.replay_memory.episodes])
            mean_val = mean([e.val.mean() for e in self.replay_memory.episodes])
            mean_clip_frac = mean([pd['cf'] for pd in pi_infos])
            opt_end_time = time.time()
            log_data = {
                **log_data,
                'n_minibatch_steps': n_minibatches,
                'mean_buffer_logp': mean([x.logp.mean() for x in self.replay_memory.episodes]),
                'mean_critic_loss': mean(critic_losses),
                'mean_policy_loss': mean(policy_losses),
                'mean_reconstruction_loss': mean(reconstruction_losses),
                'mean_entropy_loss': mean(entropy_losses),
                'mean_val': mean_val,
                'mean_pi_clip_frac': mean_clip_frac,
                'n_hidden_updates': n_hidden_updates,
                'kl_final_minibatch': final_minibatch_kl,
                'kl_final': kl_after,
                'used_backup': 1*used_backup,
                'update_time_s': opt_end_time - opt_start_time
            }
                

            print(f"Update {1+self.counts['updates']}: {n_minibatches} iters, {n_hidden_updates} hidden updates.")
            print(f" > {len(self.replay_memory.episodes)} episodes since last update.")
            print(f" > Total steps: {steps.sum()} - avg {steps.mean():.2f} for {len(steps)} eps.")
            print(f" > Mean reward: {mean([x.rew.sum() for x in self.replay_memory.episodes]):.2f}")
            print(f" > Mean logp: {mean([x.logp.mean() for x in self.replay_memory.episodes]):.2f}")
            print(f" > Mean reconstruction loss: {np.mean(reconstruction_losses):.4f}")
            print(f" > Mean critic loss: {mean(critic_losses):.2f}")
            print(f" > Mean val {mean_val:.4f}")
            print(f" > Mean pi clip frac: {mean_clip_frac:.2f}")
            print(f" > KL est: {terminal_kl:.3f} -->  {kl_after:.3f}")

            self.replay_memory.clear()
            self.counts['updates'] += 1
        
        self.log('update_data', log_data)

    def clear_memory(self):
        self.replay_memory.clear()
        self.last_update_episode = self.counts['episodes']

    def start_episode(self, n_parallel=None):
        self.n_parallel = n_parallel
        self.reset_hidden()
        self.reset_state()

        
        self.episode_reward_streaks = np.zeros(n_parallel) 
        self.logged_reward_counts.append(np.zeros(n_parallel))
        self.logged_rewards.append(np.zeros(n_parallel))
        self.logged_lengths.append(np.zeros(n_parallel))
        self.logged_streaks.append(np.zeros(n_parallel))

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
        episode_reward = sum(e.rew for e in self.active_episodes)
        if bool(log):
            self.log(
                'episode_data',
                {
                    'ep_no': self.counts['episodes'],
                    'ep_len': np.mean([len(e) for e in self.active_episodes]),
                    'total_reward': episode_reward,
                },
            )

        for ep in self.active_episodes:
            ep.freeze()
            self.calculate_advantages(ep)
            self.replay_memory.add_episode(ep)

        self.counts['episodes'] += len(self.active_episodes)

        if len(self.replay_memory.episodes) >= self.learning_config['batch_size']:
            self.optimize()
            self.last_update_episode = self.counts['episodes']

        self.grad_log_sync()