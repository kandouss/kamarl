import torch
import wandb
from collections import defaultdict
import time
import os
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, name, *args, key_path=[], **kwargs):
        self._args, self._kwargs = args, kwargs
        self.name = name
        if isinstance(key_path, str):
            self.key_path = [key_path]
            # self.key_path = [key_path, self.name]
        else:
            self.key_path = key_path
            # self.key_path = [*key_path, self.name]
        self.children = []
        self.data = defaultdict(list)
        self.hparams = []

    def get_data(self, recursive=True):
        base_path = "/".join(self.key_path)
        return {
            **{f"{base_path}/{k}": v for k, v in self.data.items()},
            **{k: v for c in self.children for k, v in c.get_data().items()},
        }

    def sub_logger(self, name, **kwargs):
        ret = self.__class__(name=name, *self._args, key_path=[*self.key_path, name], **{**self._kwargs, **kwargs})
        self.children.append(ret)
        return ret

    def fixup(self, value):
        if isinstance(value, dict):
            return {k: self.fixup(v) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            return value.cpu().detach().numpy()
        else:
            return value

    def log_value(self, key, val):
        self.data[key].append(self.fixup(val))

    def flush_values(self, *args, **kwargs):
        pass

    def log_hyperparams(self, hparams):
        self.hparams.append(hparams)

    def abs_key(self, key):
        return '.'.join([*self.key_path, key])

class WandbLogger(Logger):
    def __init__(self, name, project, run_logger=None, key_path=[]):
        assert isinstance(key_path, list)
        super().__init__(name, project=project, run_logger=run_logger, key_path=key_path)
        if run_logger is None:
            self.run_logger = wandb.init(self.name, project=project, tags=[name])
        else:
            self.run_logger = run_logger

        self.gradient_watchers = {}
        
    def sync(self):
        self.run_logger.log()

    
    def watch(self, module):
        module._log_hooks = []

        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                log_track_grad = log_track_init(log_freq)
                module._wandb_hook_names.append('gradients/' + prefix + name)
                self._hook_variable_gradient_stats(
                    parameter, 'gradients/' + prefix + name, log_track_grad)

        print(f"Watching mod: {mod}")
        wandb.watch(mod)

    @staticmethod
    def wandb_fix_values(val):
        if isinstance(val, dict):
            return {k: WandbLogger.wandb_fix_values(v) for k,v in val.items()}
        elif isinstance(val, (list, np.ndarray, torch.Tensor)):
            if isinstance(val, torch.Tensor):
                val = val.cpu().detach().numpy()
            if np.isnan(val).any():
                print("Tried to log a hist with nan values.")
                import pdb; pdb.set_trace()
            wandb.Histogram(val)
        else:
            return val

    def log_value(self, key, val, step=None, commit=False):
        # print("Logging!")
        if (step is None) and (commit==False):
            step = self.run_logger.step
        self.run_logger.log({self.abs_key(key):val}, step=step, commit=commit)


    def log_hyperparams(self, hparams):
        tmp = {'.'.join(self.key_path):hparams}
        print(f"Logging hparams {tmp}")
        self.run_logger.config.update(tmp)



class TensorboardLogger:
    def __init__(self, project, name='base', base_dir='/fast/tb_kamarl/'):
        self.name = name
        if isinstance(project, str):
            self.key_path = []
            self.writer = SummaryWriter(log_dir=self.get_log_dir(base_dir, project))
            self._step = 0
        elif isinstance(project, self.__class__):
            self.key_path = [*project.key_path, name]
            self.writer = project.writer
            self.parent = project

        self.tmp_values = defaultdict(dict)

    @property
    def global_step(self):
        if hasattr(self, 'parent'):
            return self.parent.global_step
        else:
            return self._step
    
    @global_step.setter
    def global_step(self, global_step):
        if hasattr(self, 'parent'):
            self.parent.global_step = global_step
        else:
            self._step = global_step

    def sub_logger(self, name):
        return self.__class__(self, name)

    def abs_key(self, key):
        return '/'.join([*self.key_path, key])

    def watch(self, mod):
        pass
        
    def fix_val(self, value):
        if isinstance(value, dict):
            return {k: self.fix_val(v) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if isinstance(value, np.ndarray) and np.prod(value.shape)<=1:
            return value.item()
        else:
            return value

    def get_log_dir(self, base_dir, project):
        for suffix in map(lambda s: f'{s:04d}', itertools.count()):
            log_dir = os.path.join(base_dir, project, suffix)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                return log_dir
    
    def log_value(self, key, val, step=None, sibling_log=True, **kwargs):
        if step is None:
            step = self.global_step
        if isinstance(val, dict):
            for k,v in val.items():
                self.log_value(
                    key=f'{key}/{k}', val=v, step=step, sibling_log=sibling_log, **kwargs
                )
        elif hasattr(self, 'parent') and sibling_log:
            self.parent.tmp_values[step].setdefault(key, {})[self.name] = val
        else:
            self.writer.add_scalar(self.abs_key(key), self.fix_val(val), global_step=step)


    def flush_values(self, step=True):
        for step_no, child_dicts in self.tmp_values.items():
            for key, child_values in child_dicts.items():
                child_values = self.fix_val(child_values)
                if hasattr(list(child_values.values())[0], '__iter__'):
                    for a,b in child_values.items():
                        try:
                            self.writer.add_histogram(f'{self.abs_key(key)}.{a}', np.array(b), global_step=step_no)
                        except:
                            import pdb; pdb.set_trace()
                else:
                    self.writer.add_scalars(self.abs_key(key), self.fix_val(child_values), global_step=step_no)
                # self.log_value(key, child_values, step=step)
        self.tmp_values = defaultdict(dict)
        if step:
            self.global_step += 1

    def fix_hparams(self, hparams, pfx=[]):
        res = {}
        for k,v in hparams.items():
            if isinstance(v, dict):
                res = {**res, **self.fix_hparams(v, pfx=[*pfx, k])}
            else:
                res['.'.join([*pfx, k])] = self.fix_val(hparams)
        return res
    def log_hyperparams(self, hparams):
        try:
            # import pdb; pdb.set_trace()
            self.writer.add_hparams(self.fix_hparams(hparams), {})
        except:
            print(hparams)
            
