import torch
from torch import nn

class Module(nn.Module):
    
    def __init__(self, prev_modules, out_shapes, inner_mod):
        super().__init__()
        self.prev_modules = prev_modules
        for module in prev_modules:
            assert len(module.out_shapes) == 1
        self.in_shapes = [ module.out_shape for module in prev_modules ]
        self.out_shapes = out_shapes
        self.inner_mod = inner_mod
        
    @property
    def out_shape(self):
        assert len(self.out_shapes) == 1
        return self.out_shapes[0]
    
    @property
    def in_shape(self):
        assert len(self.in_shapes) == 1
        return self.in_shapes[0]
    
    def __call__(self, *args, **kwargs):
        assert self.inner_mod is not None
        return self.inner_mod(*args, **kwargs)
    
    def call_with_inputs(self, input_mods, inputs):
        args = []
        for prev in self.prev_modules:
            args.append(prev.call_with_inputs(input_mods, inputs))
        return self(*args)
    
class Input(Module):
    
    def __init__(self, shape):
        super().__init__([], [shape], lambda x: x)
        
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
        super().__init__(prev_modules, [tuple(out_shape)], lambda *args: torch.cat(args, axis))
    
class Linear(Module):
    
    def __init__(self, prev_module, feats, *args, **kwargs):
        assert len(prev_module.out_shape) == 2
        batch, in_feats = prev_module.out_shape
        out_shape = (batch, feats)
        super().__init__([prev_module], [out_shape], nn.Linear(in_feats, feats, *args, **kwargs))
        
class Model(Module):
    
    def __init__(self, in_mods, out_mods):
        out_shapes = [ out.out_shape for out in out_mods ]
        super().__init__(in_mods, out_shapes, None)
        self.out_mods = out_mods
        
    def __call__(self, inputs):
        return [ out.call_with_inputs(self.prev_modules, inputs) for out in self.out_mods ]
