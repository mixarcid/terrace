from copy import deepcopy
from typing import Tuple, Union
import functools
import torch

from torch.utils._pytree import tree_map

# https://pytorch.org/docs/stable/notes/extending.html

# https://colab.research.google.com/drive/1MLeSCMrc6a5Yf_6uAh0bH-IPf5c_BTQf#scrollTo=dM0sl2X6epBX

NumClassesType = Union[int, Tuple[int, ...]]

def _get_end_dim(tensor):
    if len(tensor.shape) == 0:
        return 1
    else:
        return tensor.shape[-1]

HANDLED_FUNCTIONS = {}
class CategoricalTensor(torch.Tensor):
    """ Subclass of torch Tensors with ``dtype`` ``long``. They have an additional
    num_classes member that is either an int of a tuple. If num_classes
    is an int, it is the number of classes in the tensor as whole (that is,
    all numbers in the tensor are in the range [0, num_classes)), If num_classes
    is a tuple, is is the number of classes along the last dimension of the tensor.
    For instance, if ``t.num_classes`` is ``(4,8)``, then ``t[...0]`` has 4 classes
    and ``t[...,1]`` has 8 classes. """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, num_classes: NumClassesType):
        return torch.Tensor._make_subclass(cls, tensor.to('meta'))

    def __init__(self, tensor, num_classes: NumClassesType):
        assert tensor.dtype == torch.long
        if isinstance(num_classes, int):
            num_classes = tuple([num_classes]*_get_end_dim(tensor))
        assert len(num_classes) == _get_end_dim(tensor)
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
        # print(func)
        raise NotImplementedError(func)

class NoStackCatTensor(CategoricalTensor):
    pass


def _implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

@_implements(torch.Tensor.__deepcopy__)
def _ct_deepcopy(ct, memo):
    return CategoricalTensor(deepcopy(ct.tensor, memo), deepcopy(ct.num_classes, memo))

@_implements(torch.split)
def _split(ct, *args, **kwargs):
    """ No error checking for now"""
    ret = torch.split(ct.tensor, *args, **kwargs)
    return [ CategoricalTensor(t, ct.num_classes) for t in ret ]

@_implements(torch.Tensor.__len__)
def _cat_tensor_len(ct):
    return len(ct.tensor)

@_implements(torch.Tensor.is_sparse.__get__)
def _cat_is_sparse(ct):
    return ct.tensor.is_sparse

@_implements(torch.Tensor.storage)
def _cat_storage(ct):
    return ct.tensor.storage()

@_implements(torch.Tensor.element_size)
def _cat_element_size(ct):
    return ct.tensor.element_size()

@_implements(torch.Tensor.size)
def _cat_size(ct, *args, **kwargs):
    return ct.tensor.size(*args, **kwargs)
    
@_implements(torch.Tensor.numel)
def _cat_numel(ct):
    return ct.tensor.numel()

@_implements(torch.Tensor.cuda)
def _cuda(ct):
    return CategoricalTensor(ct.tensor.cuda(), num_classes=ct.num_classes)

@_implements(torch.Tensor.cpu)
def _cpu(ct):
    return CategoricalTensor(ct.tensor.cpu(), num_classes=ct.num_classes)

@_implements(torch.Tensor.to)
def _to(ct, device):
    return CategoricalTensor(ct.tensor.to(device), num_classes=ct.num_classes)

def _construct_cat_tensor(*args):
    num_classes, (tensor_func, tensor_args) = args
    tensor = tensor_func(*tensor_args)
    return CategoricalTensor(tensor, num_classes=num_classes)

@_implements(torch.Tensor.__reduce_ex__)
def _cat_reduce_ex(self, proto):
    return (_construct_cat_tensor, (self.num_classes, self.tensor.__reduce_ex__(proto)))

@_implements(torch.cat)
def _cat(inputs: Tuple[CategoricalTensor, ...], dim: int = -1) -> CategoricalTensor:
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

@_implements(torch.stack)
def _stack(inputs: Tuple[CategoricalTensor, ...]):
    assert len(inputs) > 0
    num_classes = inputs[0].num_classes
    for t in inputs:
        assert t.num_classes == num_classes
    return CategoricalTensor(torch.stack([t.tensor for t in inputs]), num_classes)

@_implements(torch.Tensor.__getitem__)
def _getitem(t: CategoricalTensor, idx: int):
    num_classes = t.num_classes
    if isinstance(idx, tuple):
        if len(idx) == len(t.shape):
            num_classes = t.num_classes[idx[-1]]
    else:
        if len(t.shape) == 1:
            num_classes = t.num_classes[idx]
    if isinstance(num_classes, int):
        num_classes = (num_classes,)
    return CategoricalTensor(t.tensor[idx], num_classes)

@_implements(torch.Tensor.shape.__get__)
def _get_shape(t: CategoricalTensor):
    return t.tensor.shape

@_implements(torch.Tensor.device.__get__)
def _get_device(t: CategoricalTensor):
    return t.tensor.device

@_implements(torch.Tensor.dtype.__get__)
def _get_dtype(t: CategoricalTensor):
    return t.tensor.dtype