from typing import Tuple, Union
import functools
import torch

from torch.utils._pytree import tree_map

# https://pytorch.org/docs/stable/notes/extending.html

# https://colab.research.google.com/drive/1MLeSCMrc6a5Yf_6uAh0bH-IPf5c_BTQf#scrollTo=dM0sl2X6epBX


HANDLED_FUNCTIONS = {}
# assume max values apply to the last axis of the tensor
class CategoricalTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor, max_values: Tuple[int]):
        return torch.Tensor._make_subclass(cls, tensor.to('meta'))

    def __init__(self, tensor, max_values: Tuple[int]):
        assert tensor.dtype == torch.long
        assert len(max_values) == tensor.size(-1)
        self.tensor = tensor
        self.max_values = max_values

    def __repr__(self):
        return f"{self.__class__.__name__}(tensor={self.tensor}, max_values={self.max_values})"

    # Only difference with above is to add this function.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        raise NotImplementedError()

def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

@implements(torch.cat)
def cat(inputs: Tuple[CategoricalTensor, ...], axis: int = -1) -> CategoricalTensor:
    
    assert axis == -1 # for now
    tensor = torch.cat([t.tensor for t in inputs], axis)
    max_values = []
    for t in inputs:
        max_values += t.max_values
    return CategoricalTensor(tensor, max_values)

@implements(torch.stack)
def stack(inputs: Tuple[CategoricalTensor, ...]):
    assert len(inputs) > 0
    max_values = inputs[0].max_values
    for t in inputs:
        assert t.max_values == max_values
    return CategoricalTensor(torch.stack([t.tensor for t in inputs]), max_values)

@implements(torch.Tensor.__getitem__)
def getitem(t: CategoricalTensor, idx: int):
    max_values = t.max_values
    if isinstance(idx, tuple):
        if len(idx) == len(t.shape):
            max_values = t.max_values[idx[-1]]
    else:
        if len(t.shape) == 1:
            max_values = t.max_values[idx]
    if isinstance(max_values, int):
        max_values = [ max_values ]
    return CategoricalTensor(t.tensor[idx], max_values)

@implements(torch.Tensor.shape.__get__)
def get_shape(t: CategoricalTensor):
    return t.tensor.shape

@implements(torch.Tensor.device.__get__)
def get_device(t: CategoricalTensor):
    return t.tensor.device

@implements(torch.Tensor.dtype.__get__)
def get_dtype(t: CategoricalTensor):
    return t.tensor.dtype