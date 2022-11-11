import torch
import torch.nn as nn

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

class LazyLinear(Module):
    """ torch >=1.13 has this, but wanted to try my hand at implimenting it myself
    (and ensuring that older torch versions work) """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        in_feats = x.shape[-1]
        return self.make(nn.Linear, in_feats, *self.args, **self.kwargs)(x)
