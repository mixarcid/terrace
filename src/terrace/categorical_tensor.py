from copy import deepcopy
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

    def __getstate__(self):
        return (self.tensor, self.num_classes)

    def __setstate__(self, state):
        self.tensor, self.num_classes = state

    # Only difference with above is to add this function.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        # return func(*args, **kwargs)
        raise NotImplementedError(func)

class NoStackCatTensor(CategoricalTensor):
    pass


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

@implements(torch.Tensor.__deepcopy__)
def ct_deepcopy(ct, memo):
    return CategoricalTensor(deepcopy(ct.tensor, memo), deepcopy(ct.num_classes, memo))

@implements(torch.split)
def split(ct, *args, **kwargs):
    """ No error checking for now"""
    ret = torch.split(ct.tensor, *args, **kwargs)
    return [ CategoricalTensor(t, ct.num_classes) for t in ret ]

@implements(torch.Tensor.__len__)
def cat_tensor_len(ct):
    return len(ct.tensor)

@implements(torch.Tensor.cuda)
def cuda(ct):
    return CategoricalTensor(ct.tensor.cuda(), num_classes=ct.num_classes)

@implements(torch.Tensor.cpu)
def cpu(ct):
    return CategoricalTensor(ct.tensor.cpu(), num_classes=ct.num_classes)

@implements(torch.Tensor.to)
def to(ct, device):
    return CategoricalTensor(ct.tensor.to(device), num_classes=ct.num_classes)

def construct_cat_tensor(*args):
    num_classes, (tensor_func, tensor_args) = args
    tensor = tensor_func(*tensor_args)
    return CategoricalTensor(tensor, num_classes=num_classes)

@implements(torch.Tensor.__reduce_ex__)
def cat_reduce_ex(self, proto):
    return (construct_cat_tensor, (self.num_classes, self.tensor.__reduce_ex__(proto)))

@implements(torch.cat)
def cat(inputs: Tuple[CategoricalTensor, ...], dim: int = -1) -> CategoricalTensor:
    if dim < 0:
        dim = len(inputs[0].shape) - dim
    if dim == len(inputs[0].shape):
        num_classes = []
        for t in inputs:
            num_classes += t.num_classes
    else:
        num_classes = inputs[0].num_classes
        for inp in inputs:
            assert inp.num_classes == num_classes
    tensor = torch.cat([t.tensor for t in inputs], dim)
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