import numpy as np
import torch
import tqdm
from collections import namedtuple, defaultdict
import random
import gc
import numba
import pickle
import gym

def init_array(spec, length, array_hook=np.zeros):
    if isinstance(spec, gym.spaces.Box):
        return array_hook(shape=(length, *spec.shape), dtype=spec.dtype)
    elif isinstance(spec, gym.spaces.Discrete):
        return array_hook(shape=(length,), dtype=np.dtype('int'))
    elif isinstance(spec, gym.Space):
        raise ValueError("Unsupported gym space.")
    elif isinstance(spec, tuple):
        shape, dtype = spec
        return array_hook(shape=(length, *shape), dtype=dtype)
    else:
        raise ValueError

class Episode:
    def __init__(self, spaces, max_length=1000):
        super().__init__()
        self.max_length = max_length
        self.length = 0
        self.buffers = {k: init_array(v, length=max_length) for k,v in spaces.items()}
        self.buffer_keys = list(self.buffers.keys())
        self.frozen = False

    def __len__(self):
        return self.length

    def __getattr__(self, x):
        return self.buffers[x][:self.length]

    def append(self, data):
        if self.frozen:
            raise ValueError("Can't append to frozen episode.")
        for k,v in data.items():
            self.buffers[k][self.length] = v
        self.length += 1

    def freeze(self):
        for k,v in self.buffers.items():
            v.resize((self.length, *v.shape[1:]), refcheck=False) # kinda sketchy
        self.frozen = True

    def _get_indices(self, ix):
        if isinstance(ix, tuple) and len(ix)==2 and not isinstance(ix[1], str):
            ind_keys, ind_steps = ix
        elif (isinstance(ix, tuple) and all(isinstance(x, str) for x in ix)) or isinstance(ix, str):
            ind_keys, ind_steps = ix, slice(None)
        elif isinstance(ix, (slice, int)):
            ind_keys, ind_steps = None, ix
        else:
            raise ValueError
        if ind_keys is None or ind_keys == slice(None):
            ind_keys = self.buffer_keys
        if isinstance(ind_steps, slice):
            ind_steps = slice(*ind_steps.indices(self.length))

        return ind_keys, ind_steps

    def __getitem__(self, ix):
        ind_keys, ind_steps = self._get_indices(ix)
        if not isinstance(ind_keys, (list, tuple)):
            return self.buffers[ind_keys][ind_steps]

        return {k: self.buffers[k][ind_steps] for k in ind_keys}

    def __setitem__(self, ix, val):
        ind_keys, ind_steps = self._get_indices(ix)
        if not isinstance(ind_keys, (list, tuple)):
            self.buffers[ind_keys][ind_steps] = val

        else:
            for k in ind_keys:
                self.buffers[k][ind_steps] = val[k]

tmp = Episode({'hi': ((), 'float32'),
'mom': ((3,2),'uint8')})
for k in range(10):
    tmp.append({'hi': k})

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def pad_to_length(A, target_length, axis=0):
    if target_length == A.shape[axis]:
        return A
    pad_width = [(0,0)]*len(A.shape)
    pad_width[axis] = (0, int(target_length - A.shape[axis]))
    return np.pad(A, pad_width=tuple(pad_width), mode='constant', constant_values=0)

class RecurrentReplayMemory:
    def __init__(
        self,
        spaces,
        max_episode_length=1000,
        max_num_steps=100000,
        max_num_episodes=None,
    ):

        if (max_num_steps is None) == (max_num_episodes is None):
            raise ValueError(
                "Exactly one of max_steps and max_num_episodes must be specified."
            )
        self.max_num_episodes = max_num_episodes
        self.max_num_steps = max_num_steps
        self.max_episode_length = max_episode_length

        self.episodes = []#Episode(spaces, max_length=max_episode_length)]
        self.spaces = spaces

    @property
    def current_episode(self):
        return self.episodes[-1]

    @property
    def episode_lengths(self):
        return np.array([e.length for e in self.episodes], dtype=np.int)

    def __length__(self):
        return self.episode_lengths.sum()

    @property
    def full(self):
        if self.max_num_steps is not None:
            return len(self) >= self.max_num_steps
        else:
            return len(self.episodes) >= self.max_num_episodes

    def clear(self):
        del self.episodes
        gc.collect()
        self.episodes = []

    def start_episode(self):
        self.episodes.append(Episode(self.spaces, max_length=self.max_episode_length))

    def remove_an_episode(self, cmp=None):
        if len(self.episodes) == 1:
            raise ValueError("Attempting to delete only episode, but only episode might be active!")

        if cmp is None:
            del self.episodes[0]
            return []
        else:
            assert len(cmp) == len(self.episodes)
            ind = np.argmin(cmp[:-1])
            del self.episodes[ind]
            del cmp[ind]
            print(f"Removing episode {ind}/{len(cmp)}.")
            return cmp

    def end_episode(self, drop_key=None):
        self.current_episode.freeze()

        if drop_key is not None:
            drop_vals = [ep[drop_key].mean() for ep in self.episodes]
            while self.full:
                drop_vals = self.remove_an_episode(drop_vals)
        else:
            while self.full:
                self.remove_an_episode()

    def get_obs(self, X):
        if self.n_obs is None:
            return X[0]
        else:
            return X[: self.n_obs]

    @property
    def n_episodes(self):
        return len(self.episodes)

    def __len__(self):
        """
        Total number of steps in all episodes in the buffer.
        """
        return sum([len(ep) for ep in self.episodes], 0)

    def sample_sequence(
        self,
        batch_size,
        seq_len,
        collate=True,
        include_current=True,
        return_indices=False,
        equal_weight_episodes=False,
        through_end=True,
        priority_key=None,
    ):
        # through_end==True <=> we're OK with sampling sequences that end after sequences terminate.
        subtract_len = 1 if through_end else (1 + seq_len)

        if include_current:
            episodes_to_sample = self.episodes
            episode_lengths = self.episode_lengths
        else:
            episodes_to_sample = self.episodes[:-1]
            episode_lengths = self.episode_lengths[:-1]

        if equal_weight_episodes:
            episode_sample_weights = ((episode_lengths - subtract_len)>0).astype(float)
        elif priority_key is not None:
            episode_sample_weights = np.array([e[priority_key, :len(e)].sum() for e, l in episodes_to_sample])
        else:
            episode_sample_weights = (episode_lengths - subtract_len).clip(0)
        
        episode_sample_weights[episode_lengths<=(subtract_len)] = 0

        if episode_sample_weights.sum() == 0:
            return []
        
        to_sample = np.random.choice(
            len(episodes_to_sample),
            size=batch_size,
            replace=True,
            p=episode_sample_weights/episode_sample_weights.sum(),
        )

        if priority_key is None:
            sample_start_ixs = [
                np.random.choice(episode_lengths[ix] - subtract_len) for ix in to_sample
            ]
        else:
            norm = lambda A: A/A.sum()
            sample_start_ixs = [
                np.random.choice(
                    episode_lengths[ix] - subtract_len, 
                    p=norm(episodes_to_sample[ix][priority_key, :len(episodes_to_sample[ix])-subtract_len])
                ) for ix in to_sample
            ]
        end_ixs = []

        res = defaultdict(list)
        for ep_ix, start_ix in zip(to_sample, sample_start_ixs):
            end_ix = min(start_ix + seq_len, len(episodes_to_sample[ep_ix]))
            end_ixs.append(end_ix)
            for k,v in episodes_to_sample[ep_ix][start_ix:end_ix].items():
                res[k].append(pad_to_length(v, seq_len))

        if collate:
            res = {k: np.stack(v) for k,v in res.items()}

        if return_indices:
            return res, np.array([to_sample, sample_start_ixs, end_ixs]).T
        else:
            return res

    # def index_of_space(self, key):
    #     if self.spaces is None:
    #         if key == 'hidden':
    #             return -1

    # def sample_transitions(
    #     self, batch_size,
    # ):
    #     transitions = self.sample_sequence(batch_size, 2, collate=False)
    #     return transitions[0], self.get_obs(transitions[1])

    # def update_hidden(self, hidden_hook, seq_len=1000, hidden_ix=4, return_err=False):
    #     errlist = []
    #     for episode, ep_len in tqdm.tqdm(
    #         zip(self.episodes, self.episode_lengths),
    #         desc=f"Updating hidden states",
    #         total=len(self.episodes),
    #     ):
    #         if ep_len < 2:
    #             continue
    #         hx = None
    #         this_err = 0
    #         this_norm = 0
    #         counts = 0
    #         for start_ix in range(0, ep_len, seq_len):
    #             if hx is not None and isinstance(hx, tuple):
    #                 assert 0 == np.max(np.abs(episode[start_ix,-1] - hx[-1]))
    #             stop_ix = min(start_ix + seq_len, ep_len)
    #             episode_slice = episode[start_ix:stop_ix]
    #             hidden_sequence = hidden_hook(self.get_obs(episode_slice), hx)
    #             if isinstance(hidden_sequence, tuple):
    #                 hidden_sequence = np.stack(hidden_sequence, axis=1)
    #             hx = hidden_sequence[-1]

    #             adj = max(0, (stop_ix+1) - ep_len)
    #             new_values = hidden_sequence[:len(hidden_sequence)-adj]


    #             if return_err:
    #                 old_values = episode[start_ix+1:min(len(episode),stop_ix+1), hidden_ix]
    #                 this_err += (old_values*new_values).sum()
    #                 this_norm += np.array([(new_values**2).sum(), (old_values**2).sum()])
    #                 counts += np.ones(old_values.shape).sum()

    #             try:
    #                 episode[start_ix+1:min(len(episode),stop_ix+1), hidden_ix] = new_values
    #             except:
    #                 import pdb; pdb.set_trace()

    #         if return_err:
    #             errlist.append(this_err/(this_norm[0]*this_norm[1]+1.e-5)**0.5)
    #             if errlist[-1]>(1+1.e-5):
    #                 import pdb; pdb.set_trace()
    #     if np.isnan(np.mean(errlist)):
    #         import pdb; pdb.set_trace()
    #     return np.array(errlist)