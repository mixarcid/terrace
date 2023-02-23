import torch
import torch.nn as nn

from .categorical_tensor import CategoricalTensor

class Module(nn.Module):

    def __init__(self):
        super().__init__()
        self._initialized = False
        self._started_forward = False
        self._submodule_index = 0
        self._submodules = nn.ModuleList()
        self._param_index = 0
        self._params = nn.ParameterList()
        self._checkpoints = []
        self._checkpoint_index = 0

    def start_forward(self):
        # todo: these boolean values getting unweildly -- cut down!
        if self._submodule_index != 0:
            self._initialized = True
        self._submodule_index = 0
        self._param_index = 0
        self._started_forward = True
        self._checkpoint_index = 0

    def make(self, cls, *args, **kwargs):
        if not self._started_forward:
            raise RuntimeError("You must call start_forward before you call make")
        if not self._initialized:
            self._submodules.append(cls(*args, **kwargs))
        submod = self._submodules[self._submodule_index]
        self._submodule_index += 1
        return submod

    def make_param(self, cls, *args, **kwargs):
        if not self._started_forward:
            raise RuntimeError("You must call start_forward before you call make_param")
        if not self._initialized:
            self._params.append(cls(*args, **kwargs))
        param = self._params[self._param_index]
        self._param_index+= 1
        return param

    def checkpoint(self):
        if self._initialized:
            self._submodule_index, self._param_index = self._checkpoints[self._checkpoint_index]
            self._checkpoint_index += 1
        else:
            self._checkpoints.append((self._submodule_index, self._param_index))

    def is_initialized(self):
        return self._initialized or self._submodule_index > 0 or self._param_index > 0

    def parameters(self, recurse: bool = True):
        assert self.is_initialized(), "Terrace Module needs to be run on data before parameters method can be called"
        return super().parameters(recurse)

class WrapperModule(Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

class LazyLinear(WrapperModule):
    """ torch >=1.13 has this, but wanted to try my hand at implementing it myself
    (and ensuring that older torch versions work) """

    def forward(self, x):
        self.start_forward()
        in_feats = x.shape[-1]
        return self.make(nn.Linear, in_feats, *self.args, **self.kwargs)(x)


class LazyEmbedding(Module):

    def __init__(self, embedding_dims):
        super().__init__()
        self.embedding_dims = embedding_dims

    def forward(self, x: CategoricalTensor):
        self.start_forward()
        embedding_dims = [self.embedding_dims]*len(x.num_classes) if isinstance(self.embedding_dims, int) else self.embedding_dims
        ret = []
        for idx in range(x.shape[-1]):
            max_val = x.num_classes[idx]
            ret.append(self.make(nn.Embedding, max_val, embedding_dims[idx])(x.tensor[..., idx]))
        return torch.cat(ret, -1)

class LazyMultiheadAttention(WrapperModule):

    def forward(self, q, k, v):
        self.start_forward()
        embed_dim = q.shape[-1]
        self.kwargs["kdim"] = k.shape[-1]
        self.kwargs["vdim"] = v.shape[-1]
        return self.make(nn.MultiheadAttention, embed_dim, *self.args, **self.kwargs)(q, k, v)

class LazyLayerNorm(WrapperModule):

    def forward(self, x):
        self.start_forward()
        return self.make(nn.LayerNorm, x.shape[1:], *self.args, **self.kwargs)(x)
