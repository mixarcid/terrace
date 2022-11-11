from typing import *

from .typed_module import TypedModule, Linear, Embedding, MultiheadAttention, BatchNorm1d
from .comp_node import CompNode

class MakeModule:
    """ base class of an object that will generate modules
    given the type data of previous modules in the comp graph """

    def get_module(*args, **kwargs) -> TypedModule:
        raise NotImplementedError()

    # todo: these should input _types_, not compnodes directly
    def __call__(self, *args, **kwargs) -> CompNode:
        module = self.get_module(*args, **kwargs)
        return module(*args, **kwargs)

class MakeWrapper(MakeModule):

    args: List[Any]
    kwargs: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class MakeLinear(MakeWrapper):

    def get_module(self, in_node: CompNode) -> Linear:
        return Linear(in_node.shape[-1], *self.args, **self.kwargs)

class MakeBatchNorm1d(MakeWrapper):

    def get_module(self, in_node: CompNode) -> BatchNorm1d:
        return BatchNorm1d(in_node.shape[-1], *self.args, **self.kwargs)

class MakeEmbedding(MakeWrapper):

    def get_module(self, in_node: CompNode) -> Embedding:
        embed_dim = in_node.out_type_data.max_value
        return Embedding(embed_dim, *self.args, **self.kwargs)

class MakeMultiheadAttention(MakeWrapper):

    def get_module(self, qn: CompNode, kn: CompNode, vn: CompNode) -> MultiheadAttention:
        embed_dim = qn.shape[-1]
        self.kwargs["kdim"] = kn.shape[-1]
        self.kwargs["vdim"] = vn.shape[-1]
        return MultiheadAttention(embed_dim, *self.args, **self.kwargs)