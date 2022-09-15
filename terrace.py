import torch
from torch import nn

from type_data import TypeData, TensorTD

class Module(nn.Module):
    
    def __init__(self, prev_modules, out_types, inner_mod = None):
        super().__init__()
        self.prev_modules = prev_modules
        for module in prev_modules:
            assert len(module.out_types) == 1
        self.in_types = [ module.out_type for module in prev_modules ]
        self.out_types = out_types
        self.inner_mod = inner_mod
        
    @property
    def out_type(self):
        assert len(self.out_types) == 1
        return self.out_types[0]
    
    @property
    def in_type(self):
        assert len(self.in_types) == 1
        return self.in_types[0]

    @property
    def out_shapes(self):
        ret = []
        for td in self.out_types:
            assert isinstance(td, TensorTD)
            ret.append(td.shape)
        return ret
    
    @property
    def in_shapes(self):
        ret = []
        for td in self.in_types:
            assert isinstance(td, TensorTD)
            ret.append(td.shape)
        return ret

    @property
    def out_shape(self):
        assert len(self.out_types) == 1
        return self.out_shapes[0]
    
    @property
    def in_shape(self):
        assert len(self.in_types) == 1
        return self.in_shapes[0]
    
    def __call__(self, *args, **kwargs):
        if self.inner_mod is None:
            raise NotImplementedError
        return self.inner_mod(*args, **kwargs)
    
    def call_with_inputs(self, input_mods, inputs):
        """ Used by Model's __call__ method (below). Input overrides this
        recursive call for the base case """
        args = []
        for prev in self.prev_modules:
            args.append(prev.call_with_inputs(input_mods, inputs))
        return self(*args)
    
class Input(Module):
    
    def __init__(self, shape):
        super().__init__([], [TensorTD(shape)], lambda x: x)
        
    def call_with_inputs(self, input_mods, inputs):
        for mod, imp in zip(input_mods, inputs):
            if mod == self:
                return imp
        else:
            raise AssertionError
            
class Cat(Module):
    
    def __init__(self, prev_modules, axis=-1):
        in_shapes = [ mod.in_shape for mod in prev_modules ]
        out_shape = None
        for in_shape in in_shapes:
            if out_shape is None:
                out_shape = list(in_shape)
            else:
                assert len(out_shape) == len(in_shape)
                out_shape[axis] += in_shape[axis]
        super().__init__(prev_modules, [TensorTD(tuple(out_shape))], lambda *args: torch.cat(args, axis))
    
class Linear(Module):
    
    def __init__(self, prev_module, feats, *args, **kwargs):
        assert len(prev_module.out_shape) == 2
        batch, in_feats = prev_module.out_shape
        out_shape = (batch, feats)
        super().__init__([prev_module], [TensorTD(out_shape)], nn.Linear(in_feats, feats, *args, **kwargs))
        
class Model(Module):
    
    def __init__(self, in_mods, out_mods):
        out_types = [ TensorTD(out.out_shape) for out in out_mods ]
        super().__init__(in_mods, out_types)

        # we need to add all the modules in the graph to a modulelist
        # so torch knows to optimize their params
        to_explore = set(out_mods)
        my_mods = set()
        while len(to_explore) > 0:
            mod = to_explore.pop()
            my_mods.add(mod)
            to_explore = to_explore.union(set(mod.prev_modules))
        self.submodules = nn.ModuleList(my_mods)
        self.out_mods = out_mods
        
    def __call__(self, inputs):
        return [ out.call_with_inputs(self.prev_modules, inputs) for out in self.out_mods ]
