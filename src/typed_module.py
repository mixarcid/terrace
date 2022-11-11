import torch
from torch import nn
from typing import Type, Any, List, Tuple

from .type_data import TypeData, TensorTD
from .meta_utils import default_init, get_any_arg, contains_type, recursive_map, recursive_zip
from .comp_node import CompNode

def call_with_type_data(func, type_func, module, *args, **kwargs):
    """ If args contain type data, use type func, else use func.
    If comp node data, return comp node with module (which could be) """
    # arg = get_any_arg(*args, **kwargs)
    # todo: better way to determine if we're being called with a type or not?
    if contains_type(args, TypeData) or contains_type(kwargs, TypeData):
        return type_func(*args, **kwargs)
    elif contains_type(args, CompNode) or contains_type(kwargs, CompNode):
        def to_type(node):
            return node.out_type_data # if isinstance(node, TypeData) else node
        type_args = recursive_map(to_type, args)
        type_kwargs = recursive_map(to_type, kwargs)
        td = type_func(*type_args, **type_kwargs)
        return CompNode(
            args=args,
            kwargs=kwargs,
            op=module,
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
        return call_with_type_data(super().__call__, self.get_type_data, self, *args, **kwargs)

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

class Embedding(WrapperModule):

    def __init__(self, num_embed, embed_dim, *args, **kwargs):
        super(Embedding, self).__init__(nn.Embedding(num_embed, embed_dim, *args, **kwargs))
        self.embed_dim = embed_dim

    def get_shape(self, tdata):
        return tuple(list(tdata.shape) + [self.embed_dim])


class MultiheadAttention(WrapperModule):

    def get_dims(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None,
                 vdim=None, 
                 batch_first=False, 
                 device=None, 
                 dtype=None):
        kdim = embed_dim if kdim is None else kdim
        vdim = embed_dim if vdim is None else vdim
        return embed_dim, kdim, vdim

    def __init__(self, *args, **kwargs):
        super(MultiheadAttention, self).__init__(nn.MultiheadAttention(*args, **kwargs))
        embed_dim, kdim, vdim = self.get_dims(*args, **kwargs)
        self.embed_dim = embed_dim
        self.kdim = kdim
        self.vdim = vdim

    def get_type_data(self, qt, kt, vt):
        # for now, only considering unbatched case all tensors shape (L, E)
        L, Eq = qt.shape
        S, Ek = kt.shape
        shape1 = (L, self.embed_dim)
        shape2 = (L, S)
        return TensorTD(shape1), TensorTD(shape2)

class Activation(WrapperModule):

    def __init__(self, act):
        super(Activation, self).__init__(act)

    def get_type_data(self, tdata):
        return tdata

class ReLU(Activation):

    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__(nn.ReLU(*args, **kwargs))

class BatchNorm1d(Activation):

    def __init__(self, *args, **kwargs):
        super(BatchNorm1d, self).__init__(nn.BatchNorm1d(*args, **kwargs))

def wraps_function(f):
    """ Wraps a get_type_data function with f so the resulting
    function can take either type data or tensors """
    def wrapper(td_f):
        def inner(*args, **kwargs):
            return call_with_type_data(f, td_f, f, *args, **kwargs)
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

    def __init__(self, in_nodes, out_nodes):
        super(Model, self).__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        # do bfs to determine which submodules we need to register
        self.submodules = nn.ModuleList()
        to_explore = []
        recursive_map(lambda node: to_explore.append(node), out_nodes)
        while len(to_explore) > 0:
            node = to_explore.pop()
            if isinstance(node.op, nn.Module):
                self.submodules.append(node.op)
            recursive_map(lambda parent: to_explore.append(parent), node.args)
            recursive_map(lambda parent: to_explore.append(parent), node.kwargs)

    def forward(self, inputs):
        for node, input in recursive_zip(self.in_nodes, inputs):
            node.set_value(input)
        return recursive_map(lambda node: node.execute(), self.out_nodes)

    def get_type_data(self, in_td, out_td):
        return self.out_type_data

    @property
    def out_type_data(self):
        return recursive_map(lambda node: node.out_type_data, self.out_nodes)