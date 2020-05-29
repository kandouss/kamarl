import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba

import os
from collections import defaultdict

import gym
import pickle

from kamarl.modules import ConvNet, SeqLSTM, make_mlp, device_of
from kamarl.buffers import RecurrentReplayMemory
from kamarl.agents import Agent
from kamarl.utils import space_to_dict, dict_to_space, get_module_inputs

@numba.jit#(numba.float32[:](numba.float32[:], numba.float32))
def discount(rewards, gamma):
    discounted_rewards = 0*rewards
    c0 = 0.0
    ix = len(rewards)-1
    for x in rewards[::-1]:
        c0 = x + gamma * c0
        discounted_rewards[ix] = c0
        ix -= 1
    return discounted_rewards


class PPOLSTM(nn.Module):
    default_hyperparams = dict(
        conv_layers = [
            {'out_channels': 8, 'kernel_size': 3, 'stride': 3, 'padding': 0},
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        input_trunk_layers = [128],
        lstm_hidden_size = 128,
        val_mlp_layers = [128,128],
        pi_mlp_layers = [128]
    )

    def __init__(
            self,
            observation_space,
            action_space,
            hyperparams = {}
        ):
        # if not isinstance(observation_space, gym.spaces.Box):
        #     raise ValueError(
        #         f"{self.__class__.__name__} only supports box (image) observations"
        #     )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"{self.__class__.__name__} only supports discrete actions"
            )

        self.hyperparams = {
            **self.default_hyperparams,
            **hyperparams,
            'action_space': space_to_dict(action_space),
            'observation_space': space_to_dict(observation_space)
        }

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
        for c in self.hyperparams['conv_layers']:
            conv_layers.append(nn.Conv2d(in_channels, **c))
            conv_layers.append(nn.ReLU(inplace=True))
            in_channels = c['out_channels']

        self.conv_layers = ConvNet(
            *conv_layers[:-1],
            image_size=input_image_shape
        )
        self.combined_input_layers = make_mlp([self.conv_layers.n + n_flat_inputs, *self.hyperparams['input_trunk_layers']], nn.Tanh)
        # self.combined_input_layers = nn.Sequential(
        #     conv_layers,
        #     *make_mlp([conv_layers.n, *self.hyperparams['input_trunk_layers']], nn.Tanh)
        # )
        self.lstm = SeqLSTM(self.combined_input_layers[-1].out_features, self.hyperparams['lstm_hidden_size'])
        
        self.mlp_val = make_mlp(
            layer_sizes=[self.lstm.hidden_size, *self.hyperparams['val_mlp_layers'], 1],
            nonlinearity=nn.Tanh
        )

        self.mlp_pi = make_mlp(
            layer_sizes=[self.lstm.hidden_size, *self.hyperparams['pi_mlp_layers'], action_space.n],
            nonlinearity=nn.Tanh
        )

    def empty_hidden(self, numpy=False):
        if numpy:
            return np.zeros((2, self.hyperparams['lstm_hidden_size']), dtype=np.float32)
        else:
            return torch.zeros(
                (2, self.hyperparams['lstm_hidden_size']),
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
            try:
                return (
                    X['pov'], 
                    torch.cat(tuple([
                        (
                            F.one_hot(torch.tensor(X[k]), self.observation_space[k].n).float() if isinstance(self.observation_space[k], gym.spaces.Discrete)
                            else expand_dims(torch.tensor(X[k])).float()
                        )
                        for k in self.input_keys
                    ]), dim=-1).to(device_of(self))
                )
            except KeyError:
                import pdb; pdb.set_trace()
        else:
            raise ValueError(f"Can't process input of type {type(X)}")

    def pi(self, X, hx):
        X_image, X_other = self.process_input(X)
        X = torch.cat([self.conv_layers(X_image), X_other], dim=-1)
        X = self.combined_input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        pi = torch.distributions.Categorical(logits=self.mlp_pi(X))
        return pi

    def v(self, X, hx):
        X_image, X_other = self.process_input(X)
        X = torch.cat([self.conv_layers(X_image), X_other], dim=-1)
        X = self.combined_input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        return self.mlp_val(X)

    def pi_v(self, X, hx):
        X_image, X_other = self.process_input(X)
        X = torch.cat([self.conv_layers(X_image), X_other], dim=-1)
        X = self.combined_input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        pi = torch.distributions.Categorical(logits=self.mlp_pi(X))
        v = self.mlp_val(X)
        return pi, v

    def step(self, X, hx=None):
        X_image, X_other = self.process_input(X)
        X = torch.cat([self.conv_layers(X_image), X_other], dim=-1)
        # import pdb; pdb.set_trace()
        X = self.combined_input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        policy_logits = self.mlp_pi(X)
        pi = torch.distributions.Categorical(logits=policy_logits)
        act = pi.sample()
        logp = pi.log_prob(act)
        # print(X.shape)

        val = self.mlp_val(X)

        return act.cpu().numpy(), val, logp, torch.stack((X, hx))

def parallel_repeat(value, n_parallel=None):
    if n_parallel is None:
        return value
    if isinstance(value, torch.Tensor):
        return torch.repeat_interleave(value[None,...], n_parallel, dim=0)
    else:
        return np.repeat(np.array(value)[None,...], n_parallel, axis=0)

class PPOAgent(Agent):
    default_hyperparams = {
        'num_minibatches': 10,
        "max_episode_length": 1000,
        "batch_size": 25, # episodes
        "minibatch_size": 256,
        "minibatch_seq_len": 10,

        'learning_rate': 3.e-4,
        "target_kl": 0.01,
        "clamp_ratio": 0.2,
        "lambda":0.97,
        "gamma": 0.99,
        'entropy_bonus_coef': 0.01,
        'value_loss_coef': 0.5,

        "module_hyperparams": {}
    }
    save_modules = ['ac', 'optimizer']
    def __init__(self, *args, hyperparams={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparams = {**self.default_hyperparams, **hyperparams}
        # self.observation_space = observation_space
        # self.action_space = action_space
        self.ac = PPOLSTM(self.observation_space, self.action_space, hyperparams=self.hyperparams['module_hyperparams'])
        self.hyperparams['module_hyperparams'] = self.ac.hyperparams

        # self.track_gradients(self.ac)

        self.metadata = {
            **self.metadata,
            'hyperparams': self.hyperparams
        }

        self.replay_memory = RecurrentReplayMemory(
            {
                'obs': self.observation_space,
                'act': self.action_space,
                'rew': ((), "float32"),
                'done': ((), "bool"),
                "val": ((), "float32"),
                "adv": ((), "float32"),
                "ret": ((), "float32"),
                "logp": ((), "float32"),
                "hx_cx": ((2, self.ac.lstm.hidden_size), "float32")
            },
            max_episode_length=self.hyperparams["max_episode_length"]+1,
            max_num_steps=(self.hyperparams["max_episode_length"]+1) * (self.hyperparams['batch_size']+1)
        )
        self.device = torch.device('cpu')

        self.n_parallel = None

        self.reset_optimizer()
        self.reset_hidden()
        self.reset_state()
        self.counts = defaultdict(int)
        self.logged_lengths = []
        self.logged_rewards = []
        self.logged_streaks = []

    def reset_state(self):
        self.state = {
            'hx_cx': self.ac.empty_hidden(numpy=True),
            'val': 0,
            'adv': 0,
            'ret': 0,
            'logp': 0
        }
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

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr = self.hyperparams['learning_rate'])

    def action_step(self, X):
        last_hx_cx = self.hx_cx.detach().cpu().numpy()
        if self.n_parallel is not None:
            # if X.shape[0] != self.n_parallel:
                # raise ValueError(f"Expected {self.n_parallel} stacked observations; got {X.shape[0]}")
            # import pdb; pdb.set_trace()
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
            'val': v.cpu().numpy(),
            'logp': logp.cpu().numpy(),
            'hx_cx': last_hx_cx
        }

        self.counts['steps'] += 1
        self.counts['episode_steps'] += 1

        return a
    def save_step(self, obs, act, rew, done):
        """ 
        Save an environment transition.
        """

        def decollate(val, ix):
            if isinstance(val, dict):
                return {k: decollate(v, ix) for k,v in val.items()}
            else:
                return val[ix]

        # Keep track of "streaks" of positive rewards.
        rew_tmp = np.array(rew)
        rew_streak = getattr(self, 'rew_streak', np.zeros_like(rew_tmp))
        best_streak = getattr(self, 'best_streak', np.zeros_like(rew_tmp))
        rew_streak = (rew_streak + 1.*(rew_tmp>0))*(rew_tmp>=0)
        # if rew_streak.max()>0:
        #     print("...",rew_streak)
        self.best_streak = np.maximum(rew_streak, best_streak)
        self.rew_streak = rew_streak
        
        
        save_dict = {
            'obs': obs, 'act': act, 'rew': rew, 'done': done,
            **self.state
        }
        # if self.n_parallel is None:
        #     self.active_episodes[0].append(save_dict)
        # else:
        for i, ep in enumerate(self.active_episodes):
            if not ep['done',-1]:
                if self.n_parallel is None:
                    ep.append(save_dict)
                else:
                    ep.append({k: decollate(v,i) for k,v in save_dict.items()})

    def calculate_advantages(self, episode, last_val=0):
        ''' Populate advantages and returns in an episode. '''
        hp = self.hyperparams
        rew = np.append(episode.rew, last_val)
        vals = np.append(episode.val, last_val)

        deltas = rew[:-1] + hp['gamma'] * (vals[1:] - vals[:-1])
        episode['adv',:] = discount(deltas, hp['gamma']*hp['lambda'])
        episode['ret',:] = discount(rew, hp['gamma'])[:-1]
        return episode

    def normalize_advantages(self):
        advantages = np.concatenate([ep.adv for ep in self.replay_memory.episodes])
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        for ep in self.replay_memory.episodes:
            ep['adv',:] = (ep['adv',:] - adv_mean) / adv_std

    def compute_loss(self, data):
        mask = data['done'].cumsum(1).cumsum(1)<=1
        clamp_ratio = self.hyperparams['clamp_ratio']

        # policy loss
        pi, v_theta = self.ac.pi_v(data['obs'], data['hx_cx'])

        logp = pi.log_prob(data['act'])

        ratio = torch.exp(logp - data['logp'])
        clip_adv = torch.clamp(ratio, 1-clamp_ratio, 1+clamp_ratio) * data['adv']
        loss_pi = -((torch.min(ratio * data['adv'], clip_adv))*mask).mean()

        loss_val = (((v_theta.squeeze() - data['ret'])**2)*mask).mean()

        approx_kl = ((data['logp'] - logp)*mask).mean().item()

        loss_ent = - pi.entropy().mean() * self.hyperparams['entropy_bonus_coef']
        loss_val = loss_val * self.hyperparams['value_loss_coef']
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clamp_ratio) | ratio.lt(1-clamp_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        # import pdb; pdb.set_trace()
        return loss_pi, loss_val, loss_ent, pi_info

    def optimize(self):
        hp = self.hyperparams
        device = self.device
        self.normalize_advantages()

        pi_infos = []
        critic_losses = []
        policy_losses = []
        entropy_losses = []
        target_kl = self.hyperparams['target_kl']
        # import pdb; pdb.set_trace()
        tmi = defaultdict(list)
        for i in range(hp['num_minibatches']):
            tmp, indices = self.replay_memory.sample_sequence(
                    batch_size=hp["minibatch_size"], seq_len=hp["minibatch_seq_len"], return_indices=True
            )
            for k,v in tmp.items():
                if k != 'hx_cx':
                    tmi[k].append(v)
            tmi['indices'].append(indices)

            # print("Mean obs reward:", (tmp['obs']['reward'] - tmp['rew']).mean())

            # ignore hiddens after the first before sending to GPU.
            tmp['hx_cx'] = np.moveaxis(tmp['hx_cx'][:,0], -2, 0)
            def to_tensor(x):
                if isinstance(x, dict):
                    return {k: to_tensor(v) for k,v in x.items()}
                return torch.from_numpy(x).to(self.device)
            
            data = to_tensor(tmp)
            # data = {k: torch.from_numpy(v).to(self.device) for k,v in tmp.items()}
            
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()

                policy_loss, critic_loss, entropy_loss, loss_metrics = self.compute_loss(data)

                kl = loss_metrics['kl']
                if i>0 and kl > 1.5 * target_kl:
                    print(f'Early stopping at step {i} due to reaching max kl.')
                    break

                
                pi_infos.append(loss_metrics)
                (policy_loss+critic_loss+entropy_loss).backward()

                critic_losses.append(critic_loss.detach().cpu().numpy())
                policy_losses.append(policy_loss.detach().cpu().numpy())
                entropy_losses.append(entropy_loss.detach().cpu().numpy())

                self.optimizer.step()

        tmi_dir = self.LOG_DIR
        print("SAVING TOO MUCH INFORMATION... ")
        print(f"  {tmi_dir} ...", end='')
        pickle.dump(dict(tmi), open(os.path.join(tmi_dir, f'tmi_{self.counts["updates"]}.p'),'wb'))
        del tmi
        # import pdb; pdb.set_trace()
        print("DONE!")
        ## Remainder of this method is informational!
        kl_iters = i

        steps = np.array([len(e) for e in self.replay_memory.episodes])
        mean_val = np.array([e.val.mean() for e in self.replay_memory.episodes]).mean()
        mean_clip_frac = np.nanmean(np.array([pd['cf'] for pd in pi_infos]))
        log_data = {
                'update_no': self.counts['updates'],
                'ep_no': self.counts['episodes'],
                'step_no': self.counts['steps'],
                'mean_buffer_logp': np.mean([x.logp.mean() for x in self.replay_memory.episodes]),
                'mean_critic_loss': np.mean(critic_losses),
                'mean_policy_loss': np.mean(policy_losses),
                'mean_entropy_loss': np.mean(entropy_losses),
                'mean_val': mean_val,
                'mean_pi_clip_frac': mean_clip_frac,
                'kl_iters': kl_iters,
                'mean_reward_streak': np.mean(self.logged_streaks),
                'max_reward_streak': np.max(self.logged_streaks),
                'mean_reward': np.mean(self.logged_rewards),
                'mean_length': np.mean(self.logged_lengths)
            }
        self.logged_streaks = []
        self.logged_rewards = []
        self.logged_lengths = []
            
        self.log('update_data',
            log_data#, step=self.counts['updates']
        )
        
        print(f"Update {1+self.counts['updates']}")
        print(f"     ({len(self.replay_memory.episodes)} episodes since last update)")
        print(f" > Total steps: {steps.sum()} - avg {steps.mean():.2f} for {len(steps)} eps.")
        print(f" > Mean logp: {np.mean([x.logp.mean() for x in self.replay_memory.episodes]):.2f}")
        print(f" > Mean reward: {np.array([x.rew.sum() for x in self.replay_memory.episodes]).mean():.2f}")
        print(f" > Mean critic loss: {np.mean(critic_losses):.2f}")
        print(f" > Mean val {mean_val:.4f}")
        print(f" > Mean pi clip frac: {mean_clip_frac:.2f}")


        self.replay_memory.clear()
        self.counts['updates'] += 1

    def clear_memory(self):
        self.replay_memory.clear()
        self.last_update_episode = self.counts['episodes']

    def start_episode(self, n_parallel=None):
        self.n_parallel = n_parallel
        self.ep_act_hist = np.zeros(self.action_space.n,dtype='int')
        self.reset_hidden()
        self.reset_state()
        self.last_val = None

        self.active_episodes = [
            self.replay_memory.get_new_episode() for x in 
            range(1 if self.n_parallel is None else self.n_parallel)
        ]
        # self.replay_memory.start_episode()
        self.was_active = True
        self.counts['episode_steps'] = 0

    def end_episode(self):
        episode_reward = sum(e.rew for e in self.active_episodes)
        self.log(
            'episode_data',
            {
                'ep_no': self.counts['episodes'],
                'ep_len': np.mean([len(e) for e in self.active_episodes]),
                'total_reward': episode_reward,
            },
        )

        self.logged_rewards += [e.rew.sum() for e in self.active_episodes]
        self.logged_lengths += [len(e) for e in self.active_episodes]

        self.logged_streaks += [self.best_streak]
        # print(self.best_streak)
        del self.rew_streak
        del self.best_streak

        for ep in self.active_episodes:
            self.calculate_advantages(ep)
            ep.freeze()
            self.replay_memory.add_episode(ep)
        # self.replay_memory.end_episode()
        self.counts['episodes'] += len(self.active_episodes)
        self.counts['episode_step'] = 0

        # if self.counts['episodes']>0 and ((self.counts['episodes'] % self.hyperparams['batch_size'])==0):
            # if self.counts['episodes'] - getattr(self, 'last_update_episode', 0) >= self.hyperparams['batch_size']:
        if len(self.replay_memory.episodes) >= self.hyperparams['batch_size']:
            self.optimize()
            self.last_update_episode = self.counts['episodes']

        self.grad_log_sync()