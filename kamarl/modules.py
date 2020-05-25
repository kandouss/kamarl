import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compare_modules(a, b):
    tot = 0
    denom = 0
    if hasattr(a, 'parameters'):
        a = a.parameters()
    if hasattr(a, 'parameters'):
        a = a.parameters()
    for x,y in zip(a,b):
        tot += ((x.cpu()-y.cpu())**2).sum()
        denom += np.prod(x.shape)
    return tot/denom

def device_of(mod):
    return next(mod.parameters()).device

def make_mlp(layer_sizes, nonlinearity=nn.Tanh):
    layers = []
    last_size = layer_sizes[0]
    for k,size in enumerate(layer_sizes[1:]):
        if k>0:
            layers.append(nonlinearity())
        layers.append(nn.Linear(last_size, size))
        last_size = size

    return nn.Sequential(*layers)
        

class ImageReshape(nn.Module):
    """ PyTorch convolutions expect images of shape (..., ..., n_channels, width, height ), where the 
    ellipses can represent any batch/sequence dims. This module reshapes a channel-minor image to conform
    to this convention and  rescales uint8 to float if necessary.
    """

    def __init__(self, rescale=True):
        super().__init__()
        self.rescale = rescale

    def forward(self, X):
        extra_dims = len(X.shape) - 3
        if X.shape[-1] == 3:
            X = X.permute(*range(extra_dims), *(np.array([2, 0, 1]) + extra_dims))
        if self.rescale:
            return X / 255.0
        return X

class ConvNet(nn.Module):
    """ This is a replacement for nn.Sequential for convnets that adds the following features
     - Computes flattened output shape (self.n) based on provided image size
     - Flattens images after convolutions
    This makes it somewhat easier to chain convolutions and layers that expect flat inputs.
    """
    def __init__(self, *modules, image_size=(64, 64, 3)):
        super().__init__()
        self.mods = nn.Sequential(ImageReshape(), *modules)
        self.n = int(
            np.prod(self.mods(torch.zeros(1, *image_size, dtype=torch.float32)).shape)
        )

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device_of(self))

        extra_dims = X.shape[:-3]
        X = X.reshape((-1, *X.shape[-3:]))
        if len(X.shape) == 3:
            X = self.mods(X[None, :]).squeeze(0)
        else:
            X = self.mods(X)

        return X.reshape(*extra_dims, self.n)


@torch.jit.script
def lstm_forward(X, hx, weight_ih, weight_hh, bias_ih, bias_hh):
    hx = hx.unsqueeze(-2)
    
    for k, x in enumerate(X.unbind(-2)):
        hx_ = torch.stack(torch._VF.lstm_cell( x, hx[...,k,:].unbind(0),
                            weight_ih, weight_hh, bias_ih, bias_hh))
        hx = torch.cat((hx, hx_.unsqueeze(2)), dim=-2)
    return hx[:, :, 1:]


@torch.jit.script
def lstm_noseq_forward(X, hx, weight_ih, weight_hh, bias_ih, bias_hh):
    hx_out = hx[:,:,0:1,:].clone()

    for k, x in enumerate(X.unbind(-2)):
        hx_ = torch.stack(torch._VF.lstm_cell( x, hx[..., k, :].unbind(0),
                               weight_ih, weight_hh, bias_ih, bias_hh))
        hx_out = torch.cat((hx_out, hx_.unsqueeze(-2)), dim=-2)
    return hx_out[:, :, 1:]


class SeqLSTM(nn.RNNCellBase):
    """ The built-in PyTorch RNN modules only return the hidden states for the final step in an input 
    sequence. This LSTM module returns all the intermediate hidden states, and does so by looping over the
    intermediate hidden states. The actual looping is accelerated by the jit-compiled methods `lstm_forward`
    and `lstm_noseq_forward`, also defined in this file.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__(input_size=input_size, hidden_size=hidden_size, bias=True, num_chunks=4)

    def forward(self, X, hx=None, vec_hidden=False):
        try:
            out_shape = (*X.shape[:-1], self.hidden_size)
            while X.dim() < 3:
                X = X.unsqueeze(0)

            self.check_forward_input(X[:,0,:])
            
            ## TODO: remove duplicate code here.
            if bool(vec_hidden):
                if hx is None:
                    hx = torch.zeros(2, X.size(0), X.size(1), self.hidden_size, dtype=X.dtype, device=X.device)
                else:
                    assert len(hx) == 2
                    if isinstance(hx, tuple):
                        hx = torch.stack(hx)

                self.check_forward_hidden(X[:,0,:], hx[0][:,0,:], '[0]')
                self.check_forward_hidden(X[:,0,:], hx[1][:,0,:], '[1]')

                res = lstm_noseq_forward(
                    X, hx,
                    self.weight_ih, self.weight_hh,
                    self.bias_ih, self.bias_hh,
                )
            else:
                if hx is None:
                    hx = torch.zeros(2, X.size(0), self.hidden_size, dtype=X.dtype, device=X.device)
                else:
                    assert len(hx) == 2
                    if isinstance(hx, tuple):
                        hx = torch.stack(hx)
                    if hx.dim() == 2: # If the hidden state has no batch dimension, make one up
                        hx = hx.unsqueeze(1)
                    try:
                        self.check_forward_hidden(X[:,0,:], hx[0], '[0]')
                        self.check_forward_hidden(X[:,0,:], hx[1], '[1]')
                    except:
                        import pdb; pdb.set_trace()

                res = lstm_forward(
                    X, hx,
                    self.weight_ih, self.weight_hh,
                    self.bias_ih, self.bias_hh,
                )

            return res.reshape((2, *out_shape))
        except:
            import pdb; pdb.set_trace()