import torch
from torch import nn

from type_data import TypeData, TensorTD

class Module(nn.Module):
    many_inputs = False
    
    def __init__(self, prev_modules, *args, **kwargs):
        super().__init__()
        self.prev_modules = prev_modules if self.many_inputs else [ prev_modules ]
        for module in self.prev_modules:
            assert len(module.out_types) == 1
        self.in_types = [ module.out_type for module in self.prev_modules ]
        self.out_types = self.get_out_types(*args, **kwargs)
        try:
            self.submodule = self.make_submodule(*args, **kwargs)
        except NotImplementedError:
            self.submodule = None

    # user can override any of these functions, depending on what's easiest
    def get_out_types(self, *args, **kwargs):
        return [ TensorTD(shape) for shape in self.get_out_shapes(*args, **kwargs) ]

    def get_out_shapes(self, *args, **kwargs):
        return [ self.get_out_shape(*args, **kwargs) ]

    def get_out_shape(self, *args, **kwargs):
        raise NotImplementedError

    def make_submodule(self, *args, **kwargs):
        raise NotImplementedError
        
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
    
    def forward(self, *args, **kwargs):
        if self.submodule is None:
            raise NotImplementedError
        return self.submodule(*args, **kwargs)
    
    def call_with_inputs(self, input_mods, inputs):
        """ Used by Model's forward method (below). Input overrides this
        recursive call for the base case """
        args = []
        for prev in self.prev_modules:
            args.append(prev.call_with_inputs(input_mods, inputs))
        return self(*args)

# overload all operators at once
def set_operator(method):
    setattr(Module, method, lambda self, other: OpModule((self, other), method))
    
for method in [ "__add__", "__sub__", "__mul__", "__div__", "__radd__", "__rsub__", "__rmul__", "__rdiv__" ]:
    set_operator(method)

class OpModule(Module):
    
    many_inputs = True
    
    def get_out_shape(self, method_name):
        self.method_name = method_name
        return self.in_shapes[0]

    def forward(self, arg1, arg2):
        method = getattr(arg1, self.method_name)
        return method(arg2)
    
class Input(Module):
    many_inputs = True
    
    def __init__(self, shape, dtype = torch.float32):
        super().__init__([], shape, dtype)

    def forward(self, x):
        return x

    def get_out_types(self, shape, dtype):
        if isinstance(shape, tuple):
            return [ TensorTD(shape, dtype) ]
        else:
            # It's a typedata
            return [ shape ]
        
    def call_with_inputs(self, input_mods, inputs):
        for mod, imp in zip(input_mods, inputs):
            if mod == self:
                return imp
        else:
            raise AssertionError
            
class Cat(Module):
    many_inputs = True

    def get_out_shape(self, axis=-1):
        self.axis = axis
        out_shape = None
        for in_shape in self.in_shapes:
            if out_shape is None:
                out_shape = list(in_shape)
            else:
                assert len(out_shape) == len(in_shape)
        out_shape[axis] += in_shape[axis]
        return tuple(out_shape)

    def forward(self, *inputs):
        return torch.cat(inputs, axis=self.axis)
        
class Linear(Module):

    def make_submodule(self, feats, *args, **kwargs):
        batch, in_feats = self.in_shape
        return nn.Linear(in_feats, feats, *args, **kwargs)
    
    def get_out_shape(self, feats, *args, **kwargs):
        batch, in_feats = self.in_shape
        return (batch, feats)

class Conv1d(Module):

    def make_submodule(self, feats, *args, **kwargs):
        batch, in_feats, sz = self.in_shape
        return nn.Conv1d(in_feats, feats, *args, **kwargs)

    def get_out_shape(self, feats, kernel_size, stride=1, padding=0, dilation=1,):
        batch, in_feats, sz = self.in_shape
        sz_out = (sz + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1
        return (batch, feats, sz_out)

class Activation(Module):
    nn_act = None

    def make_submodule(self, *args, **kwargs):
        assert self.nn_act is not None
        return getattr(nn, self.nn_act)(*args, **kwargs)

    def get_out_shape(self, *args, **kwargs):
        return self.in_shape

# truly cursed way of dynamically generating classes for all of
# torch's acitvation functions. Messing with the globals is v bad
# practice but this would otherwise be a ton of copy-and-pasting
for nn_act in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"]:
    globals()[nn_act] = type(nn_act, (Activation,), {})
    globals()[nn_act].nn_act = nn_act
    # globals()[nn_act] = T

class Lambda(Module):
    """ So far only handles things with a single tensor input.
    Will change at some point once I rework the many_inputs api """

    def make_submodule(self, func, shape_func = lambda x: x):
        return func

    def get_out_shape(self, func, shape_func = lambda x: x):
        return shape_func(self.in_shape)
        
class Model(Module):
    many_inputs = True

    def get_out_types(self, out_mods):
        return [ out.out_type for out in out_mods ]
    
    def __init__(self, in_mods, out_mods):
        super().__init__(in_mods, out_mods)

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
        
    def forward(self, inputs):
        return [ out.call_with_inputs(self.prev_modules, inputs) for out in self.out_mods ]
