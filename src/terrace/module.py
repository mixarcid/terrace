from typing import Tuple, Union
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
        self._submodule_list = []
        self._param_index = 0
        self._params = nn.ParameterList()
        self._checkpoints = []
        self._checkpoint_index = 0

    def start_forward(self):
        # todo: these boolean values getting unweildly -- cut down!
        if self.__dict__["_submodule_index"] != 0:
            self.__dict__["_initialized"] = True
        self.__dict__["_submodule_index"] = 0
        self.__dict__["_param_index"] = 0
        self.__dict__["_started_forward"] = True
        self.__dict__["_checkpoint_index"] = 0

    def make(self, cls, *args, **kwargs):
        if not self.__dict__["_started_forward"]:
            raise RuntimeError("You must call start_forward before you call make")
        if not self.__dict__["_initialized"]:
            submod = cls(*args, **kwargs)
            self._submodules.append(submod)
            self._submodule_list.append(submod)
        submod = self.__dict__["_submodule_list"][self.__dict__["_submodule_index"]]
        self.__dict__["_submodule_index"] += 1
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

    def loop_start(self):
        raise NotImplementedError

    def loop_body(self):
        raise NotImplementedError

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
    """ LazyEmbedding uses the num_classes from CategoricalTensors 
    to determine embedding weight size. Note that it assumes tensors
    have shape (..., N) where N is the number of categorical features.
    So, in most cases where you have a batch of single categorical features,
    you must give the embedding a tensor of shape (B, 1). Admittedly
    this is a bit weird, but it does nicely extend to cases where you have
    multiple categorical features. In this case, the embedding creates
    an ``nn.Embedding`` layer for each feature and concatenates the result
    together. Thus the output of this layer has shape (B, E*N), where
    E is the ``embedding_dim``. """

    def __init__(self, embedding_dims: Union[Tuple[int], int]):
        """ If ``embedding_dims`` is a tuple, it specifies the per-feature
        embedding dimension (must have the same length as the ``num_classes``
        of the input tensor). If it's an int, we use the same dimension for
        the embedding of all the features. """
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
