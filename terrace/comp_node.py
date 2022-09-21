import torch
from torch import nn
from typing import Callable, Any, List, Dict

from .type_data import TypeData, TensorTD
from .meta_utils import default_init, recursive_map

@default_init
class CompNode:
    """ Node in the computational graph. Is defined by in nodes,
    out type data, and an operation """

    args: List[Any] # can be any container containing nodes
    kwargs: Dict[str, Any]
    op: Callable
    out_type_data: Any # can be any container containing type data

    def execute(self):
        """ Recursively excutes previous nodes to get the final value """
        exec_args = recursive_map(lambda node: node.execute(), self.args)
        exec_kwargs = recursive_map(lambda node: node.execute(), self.kwargs)
        return self.op(*exec_args, **exec_kwargs)

    @property
    def shape(self):
        return self.out_type_data.shape

    def __getitem__(self, idx):
        op = lambda node: node[idx]
        return CompNode([self], {}, op, self.out_type_data[idx])
    
class Input(CompNode):

    value: Any
    has_set_value: bool

    def __init__(self, type_data):
        super().__init__([], {}, lambda x: x, type_data)
        self.value = None
        self.has_set_value = False

    def set_value(self, value):
        self.value = value
        self.has_set_value = True

    def execute(self):
        assert self.has_set_value
        return self.value
