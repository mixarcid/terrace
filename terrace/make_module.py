from typing import *

from .typed_module import TypedModule, Linear
from .comp_node import CompNode

class MakeModule:
    """ base class of an object that will generate modules
    given the type data of previous modules in the comp graph """

    def get_module(*args, **kwargs) -> TypedModule:
        raise NotImplementedError()

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