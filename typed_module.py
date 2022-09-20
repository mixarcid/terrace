import torch
from torch import nn
from typing import Type, Any, List, Tuple

from type_data import TypeData, TensorTD
from meta_utils import default_init, get_any_arg, contains_type, recursive_map
from comp_node import CompNode

def call_with_type_data(func, type_func, *args, **kwargs):
    """ If args contain type data, use type func, else use func.
    If comp node data, return comp node with func """
    arg = get_any_arg(*args, **kwargs)
    if contains_type(arg, TypeData):
        return type_func(*args, **kwargs)
    elif contains_type(arg, CompNode):
        # todo: doesn't work. Need the in types, not the in nodes
        type_args = recursive_map(lambda node: node.out_type_data, args)
        type_kwargs = recursive_map(lambda node: node.out_type_data, kwargs)
        td = type_func(*type_args, **type_kwargs)
        return CompNode(
            args=args,
            kwargs=kwargs,
            op=func,
            out_type_data=td
        )
    else:
        return func(*args, **kwargs)
    
class TypedModule(nn.Module):

    def __init__(self):
        super(TypedModule, self).__init__()

    def get_shape(self, *args, **kwargs):
        raise NotImplementedError

    def get_shapes(self, *args, **kwargs):
        raise NotImplementedError
        
    def get_type_data(self, *args, **kwargs):
        try:
            return TensorTD(self.get_shape(*args, **kwargs))
        except NotImplementedError:
            return [ TensorTD(shape) for shape in self.get_shapes(*args, **kwargs) ]
        except NotImplementedError:
            raise
        
    def __call__(self, *args, **kwargs):
        return call_with_type_data(super().__call__, self.get_type_data, *args, **kwargs)

class WrapperModule(TypedModule):
    submodule: nn.Module

    def __init__(self, submodule: nn.Module):
        super(WrapperModule, self).__init__()
        self.submodule = submodule
    
    def forward(self, *args, **kwargs):
        return self.submodule(*args, **kwargs)

class Linear(WrapperModule):

    def __init__(self, in_feats, out_feats, *args, **kwargs):
        super(Linear, self).__init__(nn.Linear(in_feats, out_feats, *args, **kwargs))
        self.in_feats = in_feats
        self.out_feats = out_feats

    def get_shape(self, tdata):
        batch, in_feats = tdata.shape
        assert in_feats == self.in_feats
        return (batch, self.out_feats)

class Activation(WrapperModule):

    def __init__(self, act):
        super(Activation, self).__init__(act)

    def get_type_data(self, tdata):
        return tdata

class ReLU(Activation):

    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__(nn.ReLU(*args, **kwargs))

def wraps_function(f):
    """ Wraps a get_type_data function with f so the resulting
    function can take either type data or tensors """
    def wrapper(td_f):
        def inner(*args, **kwargs):
            return call_with_type_data(f, td_f)
        return inner
    return wrapper
            
@wraps_function(torch.cat)
def cat(in_types: Tuple[TensorTD, ...], axis: int) -> TensorTD:
    assert len(in_types) > 0
    out_shape = None
    for td in in_types:
        in_shape = td.shape
        if out_shape is None:
            out_shape = list(in_shape)
        else:
            assert len(out_shape) == len(in_shape)
            out_shape[axis] += in_shape[axis]
    return TensorTD(tuple(out_shape), dtype=in_types[0].dtype)


class Model(TypedModule):

    in_nodes: Any # can be any container containing Inputs
    out_nodes: Any # can be any container containing CompNodes

    # todo: optain all modules in init function so it works with autograd
    def __init__(self, in_nodes, out_nodes):
        super(Model, self).__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
    
    def forward(self, inputs):
        # todo: use a recusive zip to allow for arbitrary containers
        self.in_nodes.set_value(inputs)
        return self.out_nodes.execute()
