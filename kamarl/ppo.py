import numpy as np
import torch
import torch.nn as nn
import numba

import os
from collections import defaultdict

import gym

from kamarl.modules import ConvNet, SeqLSTM, make_mlp, device_of
from kamarl.buffers import RecurrentReplayMemory
from kamarl.agent import Agent

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
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError(
                f"{self.__class__.__name__} only supports box (image) observations"
            )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"{self.__class__.__name__} only supports discrete actions"
            )

        self.hyperparams = {
            **self.default_hyperparams,
            **hyperparams,
            'action_space': action_space.n,
            'observation_space': observation_space.shape
        }

        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        trunk = []

        in_channels = 3 # todo: infer number of image channels from observation space shape.
        conv_layers = []
        for c in self.hyperparams['conv_layers']:
            conv_layers.append(nn.Conv2d(in_channels, **c))
            conv_layers.append(nn.ReLU(inplace=True))
            in_channels = c['out_channels']

        conv_layers = ConvNet(
            *conv_layers[:-1],
            image_size=observation_space.shape
        )
        self.input_layers = nn.Sequential(
            conv_layers,
            *make_mlp([conv_layers.n, *self.hyperparams['input_trunk_layers']], nn.Tanh)
        )
        self.lstm = SeqLSTM(self.input_layers[-1].out_features, self.hyperparams['lstm_hidden_size'])
        
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

    def pi(self, X, hx):
        X = self.input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        pi = torch.distributions.Categorical(logits=self.mlp_pi(X))
        return pi

    def v(self, X, hx):
        X = self.input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        return self.mlp_val(X)

    def pi_v(self, X, hx):
        X = self.input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        pi = torch.distributions.Categorical(logits=self.mlp_pi(X))
        v = self.mlp_val(X)
        return pi, v

    def step(self, X, hx=None):
        X = self.input_layers(X)
        X, hx = self.lstm(X, hx, vec_hidden=False)
        policy_logits = self.mlp_pi(X)
        pi = torch.distributions.Categorical(logits=policy_logits)
        act = pi.sample()
        logp = pi.log_prob(act)

        val = self.mlp_val(X)

        return act.cpu().numpy(), val, logp, torch.stack((X, hx))




class PPOAgent(Agent):
    default_hyperparams = {
        'learning_rate': 3.e-4,
        'num_minibatches': 100,

        "replay_memory_size": 500000,  # steps
        "target_kl": 0.01,
        "episodes_per_batch": 25,
        "batch_size": 256,
        "batch_seq_len": 10,
        "clamp_ratio": 0.2,
        "lambda":0.97,
        "gamma": 0.99,

        "module_hyperparams": {}
    }
    save_modules = ['ac', 'optimizer']
    def __init__(self, observation_space, action_space, hyperparams={}):
        
        self.hyperparams = {**self.default_hyperparams, **hyperparams}
        self.observation_space = observation_space
        self.action_space = action_space
        self.ac = PPOLSTM(observation_space, action_space, hyperparams=self.hyperparams['module_hyperparams'])
        self.hyperparams['module_hyperparams'] = self.ac.hyperparams

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
            max_episode_length=10000,
            max_num_steps=self.hyperparams['replay_memory_size']
        )
        self.device = torch.device('cpu')

        self.reset_optimizer()
        self.reset_hidden()
        self.reset_state()
        self.counts = defaultdict(int)

    def reset_state(self):
        self.state = {
            'hx_cx': self.ac.empty_hidden(numpy=True),
            'val': 0,
            'adv': 0,
            'ret': 0,
            'logp': 0
        }

    def reset_hidden(self):
        self.hx_cx = self.ac.empty_hidden().to(self.device)

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
        a, v, logp, self.hx_cx = self.ac.step(X, self.hx_cx)

        self.state = {**self.state,
            'val': v.cpu().numpy().item(),
            'logp': logp.cpu().numpy().item(),
            'hx_cx': last_hx_cx
        }

        self.counts['steps'] += 1
        self.counts['episode_steps'] += 1

        return a

    def save_step(self, obs, act, rew, done):
        """ 
        Save an environment transition.
        """
        save_dict = {
            'obs': obs, 'act': act, 'rew': rew, 'done': done,
            **self.state
        }
        if not self.replay_memory.current_episode['done',-1]:
            self.replay_memory.current_episode.append(save_dict)

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
        # import pdb; pdb.set_trace()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clamp_ratio) | ratio.lt(1-clamp_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        # import pdb; pdb.set_trace()
        return loss_pi, loss_val, pi_info

    def optimize(self):
        hp = self.hyperparams
        device = self.device
        self.normalize_advantages()        

        pi_infos = []
        critic_losses = []
        target_kl = self.hyperparams['target_kl']
        for i in range(hp['num_minibatches']):
            tmp = self.replay_memory.sample_sequence(
                    batch_size=hp["batch_size"], seq_len=hp["batch_seq_len"]
            )

            # ignore hiddens after the first before sending to GPU.
            tmp['hx_cx'] = np.moveaxis(tmp['hx_cx'][:,0], -2, 0)

            data = {k: torch.from_numpy(v).to(self.device) for k,v in tmp.items()}
            
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()

                policy_loss, critic_loss, loss_metrics = self.compute_loss(data)

                kl = loss_metrics['kl']
                if i>0 and kl > 1.5 * target_kl:
                    print(f'Early stopping at step {i} due to reaching max kl.')
                    break

                
                pi_infos.append(loss_metrics)
                (policy_loss+critic_loss).backward()

                critic_losses.append(critic_loss.detach().cpu().numpy())

                self.optimizer.step()

        ## Remainder of this method is informational!
        kl_iters = i

        steps = np.array([len(e) for e in self.replay_memory.episodes])
        mean_val = np.array([e.val.mean() for e in self.replay_memory.episodes]).mean()
        mean_clip_frac = np.nanmean(np.array([pd['cf'] for pd in pi_infos]))
        log_data = {
                'update_no': self.counts['updates'],
                'ep_no': self.counts['episodes'],
                'mean_buffer_logp': np.mean([x.logp.mean() for x in self.replay_memory.episodes]),
                'mean_critic_loss': np.mean(critic_losses),
                'mean_val': mean_val,
                'mean_pi_clip_frac': mean_clip_frac,
                'kl_iters': kl_iters,
            }
            
        self.log('update_data',
            log_data#, step=self.counts['updates']
        )

        # import pdb; pdb.set_trace()
        
        print(f"Update {1+self.counts['updates']}")
        print(f" > Total steps: {steps.sum()} - avg {steps.mean():.2f} for {len(steps)} eps.")
        print(f" > Mean logp: {np.mean([x.logp.mean() for x in self.replay_memory.episodes]):.2f}")
        print(f" > Mean reward: {np.array([x.rew.sum() for x in self.replay_memory.episodes]).mean():.2f}")
        print(f" > Mean critic loss: {np.mean(critic_losses):.2f}")
        print(f" > Mean val {mean_val:.4f}")
        print(f" > Mean pi clip frac: {mean_clip_frac:.2f}")




        self.replay_memory.clear()
        self.counts['updates'] += 1

    def start_episode(self):
        self.ep_act_hist = np.zeros(self.action_space.n,dtype='int')
        self.reset_hidden()
        self.reset_state()
        self.last_val = None
        self.replay_memory.start_episode()
        self.was_active = True
        self.counts['episode_steps'] = 0

    def end_episode(self):
        episode_reward = self.replay_memory.current_episode.rew.sum()
        self.log(
            'episode_data',
            {
                'ep_no': self.counts['episodes'],
                'ep_len': len(self.replay_memory.current_episode),
                'total_reward': episode_reward
            },
        )

        LOG_INTERVAL=25
        if (self.counts['episodes'] % LOG_INTERVAL) == 0:
            if len(getattr(self, 'logged_rewards', [])) > 0:
                self.log(f'ep_{LOG_INTERVAL}',
                {
                    'ep_no': self.counts['episodes'],
                    'step_no': self.counts['steps'],
                    'mean_reward': np.mean(self.logged_rewards),
                    'lengths': self.logged_lengths,
                    'mean_length': np.mean(self.logged_lengths)
                })
            self.logged_rewards = []
            self.logged_lengths = []
        self.logged_rewards.append(episode_reward)
        self.logged_lengths.append(len(self.replay_memory.current_episode))

        self.calculate_advantages(self.replay_memory.current_episode)
        self.replay_memory.end_episode()


        if self.counts['episodes']>0 and ((self.counts['episodes'] % self.hyperparams['episodes_per_batch'])==0):
            self.optimize()

        self.counts['episodes'] += 1
        self.counts['episode_step'] = 0

