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
from kamarl.utils import space_to_dict, dict_to_space, get_module_inputs, chunked_iterable, discount_rewards, discount_rewards_tensor, log_lists, params_grads

from PIL import Image
import copy


def update_config_dict(base_config, new_config):
    updated_config = copy.deepcopy(base_config)
    for k,v in new_config.items():
        if k not in base_config:
            raise ValueError(f"Attempted to set new key {k} in config {base_config}.\n\tNot allowed!")
        if isinstance(base_config[k], dict) or isinstance(v, dict):
            if not (isinstance(base_config[k], dict) and isinstance(v, dict)):
                raise ValueError("Attempted to merge a dict and non-dict")
            else:
                updated_config[k] = update_config_dict(base_config[k], v)
        else:
            updated_config[k] = v
    return updated_config
        


class SACLSTM(nn.Module):
    default_config = {
        "conv_layers" : [
            {'out_channels': 8, 'kernel_size': 3, 'stride': 3, 'padding': 0},
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        ],
        'input_trunk_layers': [192,192],
        'lstm_hidden_size': 256,
        'q1_mlp_layers': [64,64],
        'q2_mlp_layers': [64,64],
        'pi_mlp_layers': [64,64],
        'fancy_init': False,
        'conv_nonlinearity': 'relu',
        'norm': False,
        'no_tanh': True,
        'deconv_layers': None,
    }
    

    def __init__(
            self,
            observation_space,
            action_space,
            config = None,
        ):
        if config is None:
            config = {}
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"{self.__class__.__name__} only supports discrete action spaces"
            )
        self.config = update_config_dict(self.default_config, config)

        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        input_image_shape, n_flat_inputs = get_module_inputs(self.observation_space)
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.input_keys = [x for x in self.observation_space.spaces.keys() if x != 'pov']
        else:
            self.input_keys = []

        trunk = []

        if self.config['conv_nonlinearity'].lower() == 'relu':
            nlin_hook = nn.ReLU
        elif  self.config['conv_nonlinearity'].lower() == 'leaky_relu':
            nlin_hook = nn.LeakyReLU
        elif  self.config['conv_nonlinearity'].lower() == 'elu':
            nlin_hook = nn.ELU
        else:
            raise ValueError("not sure what relu mode should be.")

        in_channels = 3 # todo: infer number of image channels from observation space shape.
        conv_layers = []
        for k,c in enumerate(self.config['conv_layers']):
            conv_layers.append(nn.Conv2d(in_channels, **c))
            in_channels = c['out_channels']
            if k < len(self.config['conv_layers']) - 1:
                conv_layers.append(nlin_hook(inplace=True))
                if self.config['norm']:
                    conv_layers.append(nn.BatchNorm2d(num_features=in_channels))

        if self.config['no_tanh']:
            tanh = nn.LeakyReLU
        else:
            tanh = nn.Tanh

        self.conv_layers = ConvNet(
            *conv_layers,
            input_shape=input_image_shape,
            output_nonlinearity=tanh,
        )
        self.combined_input_layers = make_mlp(
            [self.conv_layers.n + n_flat_inputs, *self.config['input_trunk_layers']],
            nonlinearity=nlin_hook,
            output_nonlinearity=tanh)
        
        tmp = [x.out_features for x in self.combined_input_layers if hasattr(x, 'out_features')]
        if len(tmp) == 0:
            feature_count = self.conv_layers.n+n_flat_inputs
        else:
            feature_count = tmp[-1]

        self.lstm = SeqLSTM(feature_count, self.config['lstm_hidden_size'])
        

        if self.config['deconv_layers'] is not None:
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
                deconv_layers.append(nlin_hook(inplace=True))
                if self.config['norm']:
                    conv_layers.append(nn.BatchNorm2d(num_features=out_channels))

        self.deconv_layers = DeconvNet(
            *deconv_layers[::-1],
            n_latent = self.config['lstm_hidden_size'] + action_space.n,
            image_size=input_image_shape
        )

        self.mlp_pi = make_mlp(
            layer_sizes=[self.lstm.hidden_size, *self.config['pi_mlp_layers'], action_space.n],
            nonlinearity=tanh,
            output_nonlinearity=None
        )

        self.mlp_q1 = make_mlp(
            layer_sizes=[self.lstm.hidden_size + action_space.n, *self.config['q1_mlp_layers'], 1],
            nonlinearity=tanh,
            output_nonlinearity=None
        )

        self.mlp_q2 = make_mlp(
            layer_sizes=[self.lstm.hidden_size + action_space.n, *self.config['q2_mlp_layers'], 1],
            nonlinearity=tanh,
            output_nonlinearity=None
        )


        if self.config['fancy_init']:
            for mod in self.modules():
                if hasattr(mod, '_init_parameters'):
                    mod._init_parameters()

        # self.temperature = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.temperature = torch.tensor(1.0, dtype=torch.float)

    def empty_hidden(self, numpy=False):
        if numpy:
            return np.zeros((2, self.config['lstm_hidden_size']), dtype=np.float32)
        else:
            return torch.zeros(
                (2, self.config['lstm_hidden_size']),
                dtype=torch.float32,
                device=device_of(self))

    def process_obs(self, X):
        if isinstance(self.observation_space, gym.spaces.Box):
            return (X,torch.tensor((), device=device_of(self)))
        elif isinstance(self.observation_space, gym.spaces.Dict):
            try:
                batch_dims = X['pov'].shape[:-3] # dumb hack.
            except:
                import pdb; pdb.set_trace()
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

    def process_action(self, act):
        return F.one_hot(act,  num_classes=self.action_space.n)

    def input_layers(self, X):
        if isinstance(X, dict) and 'obs' in X:
            X = X['obs']
        X_image, X_other = self.process_obs(X)
        X = self.conv_layers(X_image)
        if X_other is not None:
            X = torch.cat([X, X_other], dim=-1)
        return self.combined_input_layers(X)

    def compute_hidden(self, X):
        return self.lstm(self.input_layers(X))
        

    # def _double_q(self, X_obs, X_act, hx):

    def _double_q_pi(self, X_obs, X_act, hx):
        # X_obs = self.input_layers(obs)
        hx_cx_new = self.lstm(X_obs, hx, vec_hidden=False)
        X = torch.cat(
            (hx_cx_new[0], X_act),
             dim=-1
        )
        q1 = self.mlp_q1(X)
        q2 = self.mlp_q2(X)
        pi_logits = self.mlp_pi(hx_cx_new[0])
        return (
            q1,
            q2,
            torch.distributions.RelaxedOneHotCategorical(temperature=self.temperature, logits=pi_logits),
            hx_cx_new
        )
    def double_q_pi(self, obs, act, hx):
        X_obs = self.input_layers(obs)
        X_act = self.process_action(act)
        return self._double_q_pi(X_obs, X_act, hx)


    # def pi_v(self, X, hx, return_hidden=False):
    #     X = self.input_layers(X)
    #     hx_cx_new = self.lstm(X, hx, vec_hidden=False)

    #     pi = torch.distributions.Categorical(logits=self.mlp_pi(hx_cx_new[0]))
    #     v = self.mlp_val(hx_cx_new[0])
    #     if return_hidden:
    #         return pi, v, hx_cx_new
    #     else:
    #         return pi, v

    # def pi_v_rec(self, X, hx, act, return_hidden=False, input_dropout=0.0):
    #     X = self.input_layers(X)
    #     X = nn.functional.dropout(X, p=input_dropout)

    #     hx_cx_new = self.lstm(X, hx, vec_hidden=False)

    #     pi = torch.distributions.Categorical(logits=self.mlp_pi(hx_cx_new[0]))
    #     v = self.mlp_val(hx_cx_new[0])
    #     rec = self.deconv_layers(torch.cat((hx_cx_new[0],F.one_hot(act, self.action_space.n).float()),dim=-1))
    #     return pi, v, rec

    def step(self, X, hx=None):
        X = self.input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)

        policy_logits = self.mlp_pi(X)
        pi = torch.distributions.RelaxedOneHotCategorical(self.temperature, logits=policy_logits)
        act = pi.rsample().argmax(-1).detach().cpu().numpy()
        # import pdb; pdb.set_trace()
        # act = pi.rsample().argmax()
        # act = torch.distributions.Categorical(logits=policy_logits).sample()
        # import pdb; pdb.set_trace()
        # act_rs = pi.rsample()
        # logp = pi.log_prob(act_rs)

        # val = self.mlp_val(X)

        return act, torch.stack((X, hx))#act.argmax().cpu().numpy(), val, logp, torch.stack((X, hx))

def parallel_repeat(value, n_parallel=None):
    if n_parallel is None:
        return value
    if isinstance(value, torch.Tensor):
        return torch.repeat_interleave(value[None,...], n_parallel, dim=0)
    else:
        return np.repeat(np.array(value)[None,...], n_parallel, axis=0)

class SACRECAgent(Agent):
    default_learning_config = {
            'num_minibatches': 10,
            'min_num_minibatches': 1,
            "max_episode_length": 1000,
            "batch_size": 25, # episodes
            "hidden_update_interval": 20, # gradient updates!
            "hidden_update_n_parallel": 32,
            "minibatch_size": 256,
            "minibatch_seq_len": 10,

            "replay_memory_steps": 1e5,

            'optimizer': 'adam',
            'learning_rate': 3.e-4,
            "weight_decay": 0,

            # "kl_target": 0.01,
            # "kl_hard_limit": 0.03,
            # "clamp_ratio": 0.2,
            # "lambda":0.97,
            # 'entropy_bonus_coef': 0.001,
            # 'policy_loss_coef': 1.0,
            # 'value_loss_coef': 0.5,

            'alpha': 0.02,

            'reconstruction_loss_coef': 1.0,
            'reconstruction_loss_loss': 'l2',
            "gamma": 0.99,
            "polyak": 0.995,

            # "bootstrap_values": True,

            'predict_this_frame': False,

            'track_gradients': False,

            'save_test_image': None,
            'lstm_train_hidden_dropout': 0.0,
            'lstm_train_input_dropout': 0.0,
            'lstm_grad_clip': 10.0,
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

        self.learning_config = update_config_dict(self.default_learning_config, learning_config)
        
        self.ac = SACLSTM(self.observation_space, self.action_space, config=model_config)

        self.ac_target = copy.deepcopy(self.ac)
        for p in self.ac_target.parameters():
            p.requires_grad = False

        self.model_config = self.ac.config
        
        if self.learning_config['track_gradients']:
            self.track_gradients(self.ac)#.lstm)

        self.training = True

        self.replay_memory_array_specs = {
                'obs': self.observation_space,
                'act': self.action_space,
                'rew': ((), "float32"),
                'done': ((), "bool"),
                # "val": ((), "float32"),
                # "adv": ((), "float32"),
                # "ret": ((), "float32"),
                # "logp": ((), "float32"),
                "hx_cx": ((2, self.ac.lstm.hidden_size), "float32"),
                "hx_cx_target": ((2, self.ac.lstm.hidden_size), "float32")
            }

        self.replay_memory = RecurrentReplayMemory(
            self.replay_memory_array_specs,
            max_episode_length=self.learning_config["max_episode_length"]+1,
            max_num_steps=self.learning_config['replay_memory_steps'],
            # max_num_steps=(self.learning_config["max_episode_length"]+1) * (self.learning_config['batch_size']+1)
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
            'n_gradient_updates': self.counts['gradient_updates'],
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
        self.ep_stats = {
            'lengths': [],
            'rewards': [],
            'streaks': [],
            'reward_counts': [],
            'reward_streaks': []
        }
        self.ep_stats_hist = copy.deepcopy(self.ep_stats)

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
        if self.learning_config['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.ac.parameters(),
                lr = self.learning_config['learning_rate'],
                weight_decay = self.learning_config['weight_decay']
            )
        elif self.learning_config['optimizer'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.ac.parameters(),
                lr = self.learning_config['learning_rate'],
                weight_decay = self.learning_config['weight_decay']
            )

    def action_step(self, X):
        self.ac.eval()
        last_hx_cx = self.hx_cx.detach().cpu().numpy()
        if self.n_parallel is not None:
            if isinstance(X, dict):
                X = {k:v[:,None,...] for k,v in X.items()}
            else:
                X = X[:,None,...]
            tmp = self.ac.step(X, self.hx_cx.unbind(-2))
            a, hx_cx = [x.squeeze() for x in tmp]
            self.hx_cx = torch.stack(hx_cx.unbind(-2))
        else:
            a, self.hx_cx = self.ac.step(X, self.hx_cx)

        self.state = {**self.state,
            # 'val': v.detach().cpu().numpy(),
            # 'logp': logp.detach().cpu().numpy(),
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
        
        # self.episode_reward_streaks = (self.episode_reward_streaks + (rew_ >= min_streak_rew))*(rew_ >= 0)

        # self.logged_reward_counts[-1] += (rew_>=min_streak_rew)
        # self.logged_streaks[-1] = np.maximum(self.episode_reward_streaks, self.logged_streaks[-1])
        # self.logged_rewards[-1] += rew*still_going
        # self.logged_lengths[-1] += still_going

        try:
            self.ep_stats['reward_streaks'] = (self.ep_stats['reward_streaks'] + (rew_ >= min_streak_rew))*(rew_ >= 0)
            self.ep_stats['reward_counts'] += (rew_>=min_streak_rew) 
            self.ep_stats['streaks'] = np.maximum(self.ep_stats['streaks'], self.ep_stats['reward_streaks'])
            self.ep_stats['rewards'] += rew * still_going
            self.ep_stats['lengths'] += still_going
        except:
            import pdb; pdb.set_trace()


        if self.training:
            save_dict = {
                'obs': obs, 'act': act, 'rew': rew, 'done': done,
                **self.state
            }
            for i, ep in enumerate(self.active_episodes):
                # if done:
                    # import pdb; pdb.set_trace()
                if len(ep)<2 or not ep['done'][-2]: 
                    if self.n_parallel is None:
                        ep.append(save_dict)
                    else:
                        ep.append({k: decollate(v,i) for k,v in save_dict.items()})

    def refresh_hiddens(self, all_episodes, parallel=32, return_mse=False, torch_mode=False):
        # print(f"refreshing {len(all_episodes)} hiddens.")
        hidden_errs = log_lists()

        with torch.no_grad():
            update_sum_err = 0
            update_value_count = 0
            err = 0
            for episodes in chunked_iterable(sorted(all_episodes, key=lambda e: len(e)), size=parallel):
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
                new_hiddens = self.ac.compute_hidden(data)
                new_target_hiddens = self.ac_target.compute_hidden(data)

                new_hiddens = new_hiddens.permute(1, 2, 0, 3)
                new_target_hiddens = new_target_hiddens.permute(1, 2, 0, 3)

                if not torch_mode:
                    new_hiddens = new_hiddens.cpu().numpy()
                    new_target_hiddens = new_target_hiddens.cpu().numpy()
                    
                for i, ep in enumerate(episodes):
                    # err += ((new_hiddens[i, :len(ep)-1, :, :] - ep['hx_cx'][1:len(ep)])**2).sum()
                    hidden_errs.update({
                        'hx_cx': ((new_hiddens[i, :len(ep)-1, :, :] - ep['hx_cx'][1:len(ep)])**2).mean()**0.5,
                        'hx_cx_tgt': ((new_target_hiddens[i, :len(ep)-1, :, :] - ep['hx_cx_target'][1:len(ep)])**2).mean()**0.5
                    })
                    ep['hx_cx'][1:len(ep)] = new_hiddens[i, :len(ep)-1, :, :]
                    ep['hx_cx_target'][1:len(ep)] = new_target_hiddens[i, :len(ep)-1, :, :]

        return hidden_errs


    def calculate_advantages(self, episode, last_val=0):
        ''' Populate advantages and returns in an episode. '''
        hp = self.learning_config

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


    def compute_q_loss(self, data):
        hp = self.learning_config

        mask = (data['done'].cumsum(1)>0).cumsum(1)<=1
        mask2 = (data['done'].cumsum(1)>0).cumsum(1)<=0
        N = mask.sum()
        ########################
        # Q FUNCTION LOSS
        ########################
        # import pdb; pdb.set_trace()
        q1, q2, pi, hx_cx_next = self.ac.double_q_pi(data['obs'], data['act'], data['hx_cx'][:,:,0,...])
        # q1 = q1.squeeze()
        # q1 = q1.squeeze()
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            # Option 1
            # pi_samp = pi.rsample().argmax(axis=-1)
            # logp_pi = pi.logits.gather(-1, pi_samp[...,None]).squeeze()
            # pi_samp = self.ac_target.process_action(pi_samp)
            # Option 2
            pi_samp = pi.rsample()
            logp_pi = pi.log_prob(pi_samp)
            # acts = pi.
            # import pdb; pdb.set_trace()
            # Target Q-values
            x_obs = self.ac_target.input_layers(data['obs'])
            # x_act = self.ac_target.process_action(data['act'])
            # x_act = self.ac_target.process_action(a_sample)
            q1_target, q2_target, _, _ = self.ac_target._double_q_pi(
                x_obs[:,1:,...], pi_samp[:,:-1,...], data['hx_cx_target'][:,:,1,...]
            )
            q_pi_target = torch.min(q1_target, q2_target).squeeze()
            backup = data['rew'][:,:-1] + hp['gamma'] * mask2[:,:-1] * (q_pi_target - hp['alpha'] * logp_pi[:,:-1])
            # backup = data['rew'][:,:-1] + hp['gamma'] * mask[:,1:] * (q_pi_target - hp['alpha'] * logp_a_sample[:,1:])

        loss_q1 = (((q1[:,:-1].squeeze() - backup)*mask[:,:-1])**2).mean()
        loss_q2 = (((q2[:,:-1].squeeze() - backup)*mask[:,:-1])**2).mean()

        # print(f"logp_a_sample is {logp_pi.mean().detach().cpu().numpy().item():.2f}")
        # if data['rew'][:,:-1].sum()>0:
        #     import pdb; pdb.set_trace()
        # print(f">>>>>>>>>>>>>.   MeAN BACKUP IS {backup.mean().cpu().numpy().item():.2f}")

        loss_q = loss_q1 + loss_q2
        return loss_q, {'q_mean': ((q1+q2).squeeze()*mask).sum()/(N*2), 'q_tgt_mean': ((q1_target+q2_target).squeeze()*mask[:,1:,...]).sum()/(N*2)}

    def compute_pi_loss(self, data):
        hp = self.learning_config
        

        mask = (data['done'].cumsum(1)>0).cumsum(1)<=1
        mask2 = (data['done'].cumsum(1)>0).cumsum(1)<=0

        N = mask.sum()
        ########################
        # POLICY LOSS
        ########################
        q1, q2, pi, hx_cx_next = self.ac.double_q_pi(data['obs'], data['act'], data['hx_cx'][:,:,0,...])

        pi_samp = pi.rsample()
        logp_pi = pi.log_prob(pi_samp)
        # probs, acts = pi_samp.max(-1)
        # X_act = self.ac.process_action(acts)
        # logp_pi = acts.log()

        # Janky stuff...
        X = torch.cat((hx_cx_next[0], pi_samp), dim=-1)
        q1_pi = self.ac.mlp_q1(X)
        q2_pi = self.ac.mlp_q2(X)
        q_pi = torch.min(q1_pi, q2_pi).squeeze()
        # print(f"TEST: WHERE IS q2_pi less than q1_pi? +++++++++++IUAFHAILUFHAIUS++>>> FRAC {(q2_pi<q1_pi).float().mean().detach().cpu().numpy():.2f}")

        # if ((1.-1.*mask).sum())>1:
        #     import pdb; pdb.set_trace()

        # Entropy-regularized policy loss
        # import pdb; pdb.set_trace()
        # loss_pi = (hp['alpha'] * logp_pi - q_pi).mean()
        loss_pi = ((hp['alpha'] * logp_pi - q_pi) * mask)[:,:-1].sum()/mask[:,:-1].sum()

        # import pdb; pdb.set_trace()
        return loss_pi, {}

    def sample_replay_buffer(self, batch_size, seq_len):
        data, indices = self.replay_memory.sample_sequence(batch_size=batch_size, seq_len=seq_len, return_indices=True)
        #  = data[:-1], data[-1]
        # print(indices)
        
        # import pdb; pdb.set_trace()
        if self.replay_memory.episodes[0].tensor_mode:
            # data['hx_cx'] = data['hx_cx'][:,0,...].transpose(-2,0)
            data['hx_cx'] = data['hx_cx'].permute(2,0,1,3)
            data['hx_cx_target'] = data['hx_cx_target'].permute(2,0,1,3)
            return data
        else:
            # ignore hiddens after the first before sending to GPU.
            # data['hx_cx'] = np.moveaxis(data['hx_cx'][:,0,...], -2, 0)
            data['hx_cx'] = np.moveaxis(data['hx_cx'], -2, 0)
            data['hx_cx_target'] = np.moveaxis(data['hx_cx_target'], -2, 0)
            def to_tensor(x):
                if isinstance(x, dict):
                    return {k: to_tensor(v) for k,v in x.items()}
                return torch.from_numpy(x).to(self.device)
            return to_tensor(data)
                

    def optimize(self):
        info = self.ep_stats_hist
        log_data = {
            'update_no': self.counts['updates'],
            'ep_no': self.counts['episodes'],
            'step_no': self.counts['steps'],

            'gradient_update_no': self.counts['gradient_updates'],

            'mean_reward_streak': np.mean(info['streaks']),
            'max_reward_streak': np.max(info['streaks']),
            'reward_streak_15p': np.quantile(info['streaks'], 0.15),
            'reward_streak_85p': np.quantile(info['streaks'], 0.85),

            'mean_reward': np.mean(info['rewards']),
            'ep_return_mean': np.mean(info['rewards']),
            'ep_return_15p': np.quantile(info['rewards'], 0.15),
            'ep_return_85p': np.quantile(info['rewards'], 0.85),
            'ep_return_std': np.std(info['rewards']),
            'ep_return_min': np.min(info['rewards']),
            'ep_return_max': np.max(info['rewards']),

            'mean_length': np.mean(info['lengths']),

            'mean_reward_counts': np.mean(info['reward_counts']),
            'reward_counts_15p': np.quantile(info['reward_counts'], 0.15),
            'reward_counts_85p': np.quantile(info['reward_counts'], 0.85),

            'mean_streak_fraction': np.mean(np.array(info['streaks'])/np.array(info['reward_counts']).clip(1)),
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
            lstm_gradient_clip_counts = []

            n_hidden_updates = 0

            final_minibatch_kl = 0
            terminal_kl = 0

            # self.normalize_advantages()
            self.ac.train()

            n_minibatches = 0
            # for episode in self.replay_memory.episodes:
                # import pdb; pdb.set_trace()
                # episode.to_tensor(device=device)
            logdata = log_lists()
            lstm_grad_clip_count = 0
            for i in range(hp['num_minibatches']):


                if self.counts['gradient_updates'] % hp['hidden_update_interval'] == 0:
                    updata = self.refresh_hiddens(self.replay_memory.episodes)
                    print('#'*40)
                    print(f"###### Refreshing hx_cx, hx_cx_tgt for all {len(self.replay_memory.episodes)} episodes in replay buffer.")
                    print(f"######      >> hx_cx     -->  mean RMS {updata.mean('hx_cx'):.3f}")
                    print(f"######      >> hx_cx_tgt -->  mean RMS {updata.mean('hx_cx_tgt'):.3f}")

                # Sample a minibatch of data with which to compute losses/update parameters.
                minibatch_data = self.sample_replay_buffer(batch_size=hp["minibatch_size"], seq_len=hp["minibatch_seq_len"])

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()

                    # loss_pi, loss_val, loss_ent, loss_rec, pi_info
                    loss_q, q_info = self.compute_q_loss(minibatch_data)
                    loss_pi, pi_info = self.compute_pi_loss(minibatch_data)

                    logdata.update(pi_info)
                    logdata.update(q_info)
                    logdata.update({
                        'loss_q': loss_q,
                        'loss_pi': loss_pi,
                    })

                    # import pdb; pdb.set_trace()
                    # opt_log_lists['loss_q'].append(loss_q.detach().mean().cpu().numpy())
                    # opt_log_lists['loss_pi'].append(loss_pi.detach().mean().cpu().numpy())
                    # [k for k,v in params_grads(self.ac)[1].items() if v.abs().sum()>0]
                    # import pdb; pdb.set_trace()
                    # if self.counts['gradient_updates']>100:
                    #     import pdb; pdb.set_trace()
                    ( 
                        loss_q 
                        + loss_pi
                    ).backward()

                    lstm_grad_clip = 10.0
                    lstm_grad_clip_count = sum(
                        ((x.detach()**2>lstm_grad_clip).sum()
                        for x in self.ac.lstm.parameters())
                    ).float()
                    logdata.update({'lstm_grad_clip_count': lstm_grad_clip_count})
                    nn.utils.clip_grad_norm_(self.ac.lstm.parameters(), lstm_grad_clip)


                    self.optimizer.step()

                polyak = hp['polyak']
                with torch.no_grad():
                    for p, p_targ in zip(self.ac.parameters(), self.ac_target.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)

                # This will be the estimated kl from the last minibatch before potential early termination.
                # final_minibatch_kl = loss_metrics['kl']

                # Save metrics.
                # pi_infos.append({k:v for k,v in loss_metrics.items() if k != 'sample_prediction'})
                # critic_losses.append(critic_loss.detach().cpu().numpy())
                # policy_losses.append(policy_loss.detach().cpu().numpy())
                # entropy_losses.append(entropy_loss.detach().cpu().numpy())
                # reconstruction_losses.append(reconstruction_loss.detach().cpu().numpy())
                # lstm_gradient_clip_counts.append(lstm_gradient_clip_count.detach().cpu().numpy())

                # Number of minibatch iterations/gradient updates that actually took place.
                n_minibatches += 1
                self.counts['gradient_updates'] += 1

            if hp['save_test_image'] is not None:
                an_image = (loss_metrics['sample_prediction'].detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                print("About an image: ", an_image.mean(), an_image.min(), an_image.max())
                Image.fromarray(an_image).save(hp['save_test_image'])


            mean = lambda vals: np.nan if len(vals) == 0 else np.nanmean(vals) if not isinstance(vals[0], torch.Tensor) else torch.tensor(vals).mean().cpu().item()
            get_first = lambda vals: vals[0] if len(vals) > 0 else np.nan
            get_last = lambda vals: vals[-1] if len(vals) > 0 else np.nan

            steps = np.array([len(e) for e in self.replay_memory.episodes])
            

            
            opt_end_time = time.time()
            log_data = {
                **log_data,
                'n_minibatch_steps': n_minibatches,
                'update_time_s': opt_end_time - opt_start_time
            }


            if self.learning_config['track_gradients']:
                log_data['gradients'] = self._grad_stats
                

            print(f"Update {1+self.counts['updates']}: {n_minibatches} iters, {n_hidden_updates} hidden updates.")
            print(f" > {len(self.replay_memory.episodes)} episodes since last update.")
            print(f" > Total steps: {steps.sum()} - avg {steps.mean():.2f} for {len(steps)} eps.")
            print(f" > Mean reward: {mean([x.rew.sum() for x in self.replay_memory.episodes]):.2f}")
            # import pdb; pdb.set_trace()
            print(f" > Mean losses (q, pi): ({logdata.mean('loss_q'):.2f}, {logdata.mean('loss_pi'):.2f})")
            # import pdb; pdb.set_trace()
            print(f" > Mean (q, q_tgt): ({logdata.mean('q_mean'):.2f}, {logdata.mean('q_tgt_mean'):.2f})")
            print(f" > Mean lstm grad clip frac: {logdata.mean('lstm_grad_clip_count'):.2f}")
            # print(f" > Mean logp: {mean([x.logp.mean() for x in self.replay_memory.episodes]):.2f}")
            # print(f" > Mean reconstruction loss: {np.mean(reconstruction_losses):.4f}")
            # print(f" > Mean critic loss: {mean(critic_losses):.2f}")
            # print(f" > Mean val {mean_val:.4f}")
            # print(f" > Mean pi clip frac: {mean_clip_frac:.2f}")
            # print(f" > KL est: {terminal_kl:.3f} -->  {kl_after:.3f}")

            # self.replay_memory.clear()
            self.counts['updates'] += 1
        # import pdb; pdb.set_trace()
        self.log('update_data', log_data)

    def clear_memory(self):
        self.replay_memory.clear()
        self.last_update_episode = self.counts['episodes']

    def start_episode(self, n_parallel=None):
        self.n_parallel = n_parallel
        self.reset_hidden()
        self.reset_state()

        
        for k in self.ep_stats.keys():
            self.ep_stats[k] = np.zeros(n_parallel)
        # self.episode_reward_streaks = np.zeros(n_parallel) 
        # self.logged_reward_counts.append(np.zeros(n_parallel))
        # self.logged_rewards.append(np.zeros(n_parallel))
        # self.logged_lengths.append(np.zeros(n_parallel))
        # self.logged_streaks.append(np.zeros(n_parallel))

        # self.reset_info()
        self.last_val = None

        if self.training:
            self.active_episodes = [
                self.replay_memory.get_new_episode() for x in 
                range(1 if self.n_parallel is None else self.n_parallel)
            ]
        else:
            self.active_episodes = []


        self.was_active = True
        self.counts['episode_steps'] = 0

    def end_episode(self, log=False):
        
        if bool(log):
            info = self.ep_stats
            self.log(
                'episode_data',
                {
                    'ep_no': self.counts['episodes'],
                    'update_no': self.counts['updates'],
                    'step_no': self.counts['steps'],

                    'mean_reward_streak': np.mean(info['streaks']),
                    'max_reward_streak': np.max(info['streaks']),
                    'reward_streak_15p': np.quantile(info['streaks'], 0.15),
                    'reward_streak_85p': np.quantile(info['streaks'], 0.85),

                    'mean_reward': np.mean(info['rewards']),
                    'ep_return_mean': np.mean(info['rewards']),
                    'ep_return_15p': np.quantile(info['rewards'], 0.15),
                    'ep_return_85p': np.quantile(info['rewards'], 0.85),
                    'ep_return_std': np.std(info['rewards']),
                    'ep_return_min': np.min(info['rewards']),
                    'ep_return_max': np.max(info['rewards']),

                    'mean_length': np.mean(info['lengths']),

                    'mean_reward_counts': np.mean(info['reward_counts']),
                    'reward_counts_15p': np.quantile(info['reward_counts'], 0.15),
                    'reward_counts_85p': np.quantile(info['reward_counts'], 0.85),

                    'mean_streak_fraction': np.mean(np.array(info['streaks'])/np.array(info['reward_counts']).clip(1)),
                }
            )

        if self.training:        
            self.counts['episodes'] += len(self.active_episodes)
            for ep in self.active_episodes:
                ep.freeze()
                # self.calculate_advantages(ep)
                self.refresh_hiddens([ep])
                # np.copyto(ep.hx_cx_target, ep.hx_cx) # refreshing hiddens would be better but marginally more expensive.
                # improt 
                self.replay_memory.add_episode(ep)
                # print(f"Replay memory has {len(self.replay_memory.episodes)} episodes.")

        else:
            self.counts['episodes'] += 1 if self.n_parallel is None else self.n_parallel

            

        for k in self.ep_stats.keys():
            self.ep_stats_hist[k] = np.append(self.ep_stats_hist[k], self.ep_stats[k])
            self.ep_stats[k] = []
            
        if not hasattr(self, 'last_update_episode'):
            self.last_update_episode = 0
        if self.counts['episodes'] - self.last_update_episode >= self.learning_config['batch_size']:
            self.optimize()
            self.last_update_episode = self.counts['episodes']