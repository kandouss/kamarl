import os
import json
import torch

from kamarl.utils import space_to_dict, dict_to_space

class Agent:
    save_modules = []
    def set_device(self, dev):
        pass

    def action_step(self, obs):
        pass

    def save_step(self, *values, **kwvalues):
        pass

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def save(self, save_dir, force=False):
        save_dir = os.path.abspath(os.path.expanduser(save_dir))
        model_path = os.path.join(save_dir, 'model.p')
        metadata_path = os.path.join(save_dir, 'metadata.json')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for f in (model_path, metadata_path):
            if force is False and os.path.isfile(f):
                raise ValueError(f"Error saving {self.__class__.__name__}: save file \"{f}\" already exists.")

        for f in (model_path, metadata_path):
            if os.path.isfile(f):
                os.remove(f)

        metadata = {
            'class': self.__class__.__name__,
            'observation_space': space_to_dict(self.observation_space),
            'action_space': space_to_dict(self.action_space),
            'hyperparams': self.hyperparams,
        }
        print("Saving modules ", self.save_modules)
        torch.save({mod: getattr(self, mod) for mod in self.save_modules
                    }, model_path)

        json.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def load(cls, save_dir):
        print(f"Loading", cls.__name__)
        save_dir = os.path.abspath(os.path.expanduser(save_dir))
        model_path = os.path.join(save_dir, 'model.p')
        metadata_path = os.path.join(save_dir, 'metadata.json')

        metadata = json.load(open(metadata_path,'r'))
        ret = cls(
            observation_space=dict_to_space(metadata['observation_space']),
            action_space=dict_to_space(metadata['action_space']),
            hyperparams=metadata['hyperparams']
        )
        modules_dict = torch.load(model_path)
        for k,v in modules_dict.items():
            getattr(ret, k).load_state_dict(v.state_dict())
        del modules_dict
        return ret
