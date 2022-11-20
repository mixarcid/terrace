import torch
import torch.nn as nn

from .categorical_tensor import CategoricalTensor

class Module(nn.Module):

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.submodule_index = 0
        self.submodules = nn.ModuleList()

    def start_forward(self):
        if self.submodule_index != 0:
            self.initialized = True
        self.submodule_index = 0

    def make(self, cls, *args, **kwargs):
        if not self.initialized:
            self.submodules.append(cls(*args, **kwargs))
        submod = self.submodules[self.submodule_index]
        self.submodule_index += 1
        return submod

class WrapperModule(Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

class LazyLinear(WrapperModule):
    """ torch >=1.13 has this, but wanted to try my hand at implimenting it myself
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
        embedding_dims = [self.embedding_dims]*len(x.max_values) if isinstance(self.embedding_dims, int) else self.embedding_dims
        ret = []
        for idx in range(x.shape[-1]):
            max_val = x.max_values[idx]
            ret.append(self.make(nn.Embedding, max_val, embedding_dims[idx])(x.tensor[..., idx]))
        return torch.cat(ret, -1)
