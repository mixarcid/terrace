from typing import Tuple, Union
import functools
import torch

from torch.utils._pytree import tree_map

# https://pytorch.org/docs/stable/notes/extending.html

# https://colab.research.google.com/drive/1MLeSCMrc6a5Yf_6uAh0bH-IPf5c_BTQf#scrollTo=dM0sl2X6epBX

NumClassesType = Union[int, Tuple[int, ...]]

HANDLED_FUNCTIONS = {}
# if num_classes is a tuple, it decribes the number of classes along the last
# dimension of the tensor
class CategoricalTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor, num_classes: NumClassesType):
        return torch.Tensor._make_subclass(cls, tensor.to('meta'))

    def __init__(self, tensor, num_classes: NumClassesType):
        assert tensor.dtype == torch.long
        if isinstance(num_classes, int):
            num_classes = tuple([num_classes]*tensor.shape[-1])
        assert len(num_classes) == tensor.shape[-1]
        self.tensor = tensor
        self.num_classes = num_classes

    def __repr__(self):
        return f"{self.__class__.__name__}(tensor={self.tensor}, num_classes={self.num_classes})"

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
    num_classes = []
    for t in inputs:
        num_classes += t.num_classes
    return CategoricalTensor(tensor, num_classes)

@implements(torch.stack)
def stack(inputs: Tuple[CategoricalTensor, ...]):
    assert len(inputs) > 0
    num_classes = inputs[0].num_classes
    for t in inputs:
        assert t.num_classes == num_classes
    return CategoricalTensor(torch.stack([t.tensor for t in inputs]), num_classes)

@implements(torch.Tensor.__getitem__)
def getitem(t: CategoricalTensor, idx: int):
    num_classes = t.num_classes
    if isinstance(idx, tuple):
        if len(idx) == len(t.shape):
            num_classes = t.num_classes[idx[-1]]
    else:
        if len(t.shape) == 1:
            num_classes = t.num_classes[idx]
    if isinstance(num_classes, int):
        num_classes = [ num_classes ]
    return CategoricalTensor(t.tensor[idx], num_classes)

@implements(torch.Tensor.shape.__get__)
def get_shape(t: CategoricalTensor):
    return t.tensor.shape

@implements(torch.Tensor.device.__get__)
def get_device(t: CategoricalTensor):
    return t.tensor.device

@implements(torch.Tensor.dtype.__get__)
def get_dtype(t: CategoricalTensor):
    return t.tensor.dtype