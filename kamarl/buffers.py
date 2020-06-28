import numpy as np
import torch
import tqdm
from collections import namedtuple, defaultdict
import random
import gc
import numba
import pickle
import gym
import itertools

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def init_array(spec, length, array_hook=np.zeros):
    if isinstance(spec, gym.spaces.Box):
        return array_hook(shape=(length, *spec.shape), dtype=spec.dtype)
    elif isinstance(spec, gym.spaces.Discrete):
        return array_hook(shape=(length,), dtype=np.int)
    elif isinstance(spec, gym.Space):
        raise ValueError("Unsupported gym space.")
    elif isinstance(spec, tuple):
        shape, dtype = spec
        return array_hook(shape=(length, *shape), dtype=dtype)
    else:
        raise ValueError

def init_array_recursive(spec, length, key_list = [], array_hook=np.zeros, array_kwargs = {}):
    def dtype_fun(d):
        if array_hook in (torch.zeros,):
            return getattr(torch, np.dtype(d).name)
        else:
            return d

    if not isinstance(length, (tuple, list)):
        length = (length,)
    if isinstance(spec, gym.spaces.Box):
        return array_hook((*length, *spec.shape), dtype=dtype_fun(spec.dtype), **array_kwargs), [key_list]
    elif isinstance(spec, gym.spaces.Discrete):
        return array_hook((*length,), dtype=dtype_fun(np.dtype('int')), **array_kwargs), [key_list]
    elif isinstance(spec, tuple):
        shape, dtype = spec
        return array_hook((*length, *shape), dtype=dtype_fun(dtype), **array_kwargs), [key_list]
    elif isinstance(spec, (dict, gym.spaces.Dict)):
        if isinstance(spec, gym.spaces.Dict):
            spec = spec.spaces
        out = {}
        leaf_keys = []
        for k,v in spec.items():
            out[k], tmp = init_array_recursive(v, length, [*key_list, k], array_hook=array_hook, array_kwargs=array_kwargs)
            leaf_keys.extend(tuple(tmp))
        return out, leaf_keys

    else:
        raise ValueError("Unsupported space {spec}.")


class Episode:
    def __init__(self, spaces, max_length=1000):
        super().__init__()
        self.max_length = max_length
        self.length = 0
        self.buffers, self.flat_keys = init_array_recursive(spaces, length=max_length)
        self.frozen = False
        self.tensor_mode = False
        self.tensor_device = None

    def __len__(self):
        return self.length

    def __getattr__(self, x):
        return self.buffers[x][:self.length]

    def append(self, data):
        if self.length >= self.max_length:
            raise ValueError(f"Didn't allocate enough space in array. Trying to append step {self.length} to episode with size {self.max_length}.")
        if self.frozen:
            raise ValueError("Can't append to frozen episode.")
        for flat_key in self.flat_keys:
            src = data
            tgt = self.buffers
            for key in flat_key[:-1]:
                if key not in src:
                    break
                src = src[key]
                tgt = tgt[key]
            else:
                if flat_key[-1] in src:
                    try:
                        tgt[flat_key[-1]][self.length] = src[flat_key[-1]]
                    except:
                        import pdb; pdb.set_trace()
        # for k,v in data.items():
        #     self.buffers[k][self.length] = v
        self.length += 1

    def _iter_buffers(self):
        for key_tuple in self.flat_keys:
            ret = self.buffers
            for k in key_tuple:
                ret = ret[k]
            yield ret

    def to_tensor(self, device):
        for key_tuple in self.flat_keys:
            ret = self.buffers
            for k in key_tuple[:-1]:
                ret = ret[k]
            ret[key_tuple[-1]] = torch.from_numpy(ret[key_tuple[-1]]).to(device=device)
        self.tensor_mode = True
        self.tensor_device = device
        return self
        

    def freeze(self):
        for buffer in self._iter_buffers():
            buffer.resize((self.length, *buffer.shape[1:]), refcheck=False)
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
            ind_keys = list(self.buffers.keys())
        if isinstance(ind_steps, slice):
            ind_steps = slice(*ind_steps.indices(self.length))
        return ind_keys, ind_steps

    def __getitem__(self, ix):
        ind_keys, ind_steps = self._get_indices(ix)
        if not isinstance(ind_keys, (list, tuple)):
            ind_keys = [ind_keys]
            return_dict = False
        else:
            return_dict = True

        ret = {}#k: None for k in ind_keys}
        for key_tuple in self.flat_keys:
            if key_tuple[0] in ind_keys:
                src = self.buffers
                tgt = ret
                for key in key_tuple[:-1]:
                    tgt = tgt.setdefault(key, {})
                    src = src[key]
                tgt[key_tuple[-1]] = src[key_tuple[-1]][ind_steps]
        
        if return_dict is False:
            return next(iter(ret.values()))

        return ret

    def __setitem__(self, ix, val):
        if isinstance(val, dict):
            raise ValueError("Current implementation doesn't support dict assignment for values stored in episodes.")
        ind_keys, ind_steps = self._get_indices(ix)
        if not isinstance(ind_keys, (list, tuple)):
            ind_keys = [ind_keys]
            return_dict = False
        else:
            return_dict = True

        for key_tuple in self.flat_keys:
            if key_tuple[0] in ind_keys:
                tgt = self.buffers
                for key in key_tuple[:-1]:
                    tgt = tgt[key]
                tgt[key_tuple[-1]][ind_steps] = val[ind_steps]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def pad_to_length(A, target_length, axis=0):
    if isinstance(A, dict):
        return {k: pad_to_length(v, target_length=target_length, axis=axis) for k,v in A.items()}
    else:
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

    def get_new_episode(self):
        return Episode(self.spaces, max_length=self.max_episode_length)

    def add_episode(self, ep):
        self.episodes.append(ep)
        while self.full:
            self.remove_an_episode()

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
        include_current=True,
        return_indices=False,
        equal_weight_episodes=False,
        through_end=True,
        priority_key=None,
        compute_hidden_hook=None,
    ):
        with torch.no_grad():
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

            hiddens = []

            # res = defaultdict(list)
            if self.episodes[0].tensor_mode:
                ret, ret_keys = init_array_recursive(
                    self.spaces, (batch_size, seq_len), 
                    array_hook=torch.zeros, array_kwargs={'device':self.episodes[0].tensor_device})
            else:
                ret, ret_keys = init_array_recursive(
                    self.spaces, (batch_size, seq_len),
                )
            sample_info = []
            for ix_in_batch, (ep_ix, start_ix) in enumerate(zip(to_sample, sample_start_ixs)):
                end_ix = min(start_ix + seq_len, len(episodes_to_sample[ep_ix]))
                end_ixs.append(end_ix)
                sample_info.append((ix_in_batch, ep_ix, start_ix, end_ix))
                tmp = episodes_to_sample[ep_ix][start_ix:end_ix]
                for key_list in ret_keys:
                    src = tmp
                    tgt = ret
                    for k in key_list[:-1]:
                        src = src[k]
                        tgt = tgt[k]
                    tgt[key_list[-1]][ix_in_batch, :end_ix-start_ix, ...] = src[key_list[-1]]
                if compute_hidden_hook is not None:
                    hiddens.append(compute_hidden_hook(episodes_to_sample[ep_ix]['obs', :start_ix]))

            if compute_hidden_hook is not None:
                ret['hx_cx'] = torch.stack(hiddens, -2)

            if return_indices:
                return ret, np.array([to_sample, sample_start_ixs, end_ixs]).T
            else:
                return ret
