import torch
from copy import deepcopy
from typing import Type, List, Tuple, Union, Dict, Type, Optional
from collections import defaultdict
from sympy import Symbol, Expr, Integer, symbols

from .meta_utils import default_init, get_type_name

def to_expr(item: Union["ShapeVar", int]) -> Expr:
    if isinstance(item, int):
        return Integer(item)
    else:
        return item.expr

class ShapeVar:
    """ If we don't know the dimension of something at graph time,
    put it in a shape var (e.g. batch) """

    expr: Expr

    def __init__(self, expr: Union[Expr, str]):
        if isinstance(expr, str):
            self.expr = symbols(expr, integer=True)
        else:
            self.expr = expr

    # for everything in this library, get_name is used when pretty-printing
    # or exporting models, to pdfs, __repr__ is used for regular printing
    def get_name(self) -> str:
        return str(self.expr)
    
    def __repr__(self) -> str:
        return f"ShapeVar[{self.get_name()}]"

    def maybe_int(self) -> Union["ShapeVar", int]:
        """ If the expression evals to an int, returns an int """
        try:
            return int(self.get_name())
        except ValueError:
            return self

    def __eq__(self, other: Union["ShapeVar", int]):
        return to_expr(other).equals(self.expr).maybe_int()

# somewhat cursed way of generating all the operators
def set_operator(method):
    def op(self, other):
        return ShapeVar(getattr(self.expr, method)(to_expr(other))).maybe_int()
    setattr(ShapeVar, method, op)
    
for method in [ "__add__", "__sub__", "__mul__", "__floordiv__", "__radd__", "__rsub__", "__rmul__", "__rfloordiv__" ]:
    set_operator(method)
    
@default_init
class TypeData:
    """ tabulates all the stuff we need to know about some future
    variable at graph creation time. E.g. the shape of a tensor """

    # the type the variable will be at runtime
    runtime_type: Type

    def __repr__(self) -> str:
        return f"TypeData[{self.get_name()}]"

    def get_name(self) -> str:
        runtime_type_name = self.get_type_name()
        try:
            return f"{runtime_type_name}[{self.get_params_repr()}]"
        except NotImplementedError:
            return runtime_type_name

    def get_type_name(self) -> str:
        return get_type_name(self.runtime_type)

    def get_params_repr(self) -> str:
        raise NotImplementedError

ShapeType = Tuple[Union[int, ShapeVar], ...]

def normalize_idx(sz, idx):
    return idx if idx >= 0 else sz + idx

class TensorTD(TypeData):
    """ Type data for tensor. This is what will be mostly used """
    shape: ShapeType
    dtype: torch.dtype
    # max_values are used to define the max values taken by long tensors
    # for classification. For everything else will be None
    # assume max values apply to the last axis of the tensor
    max_values: Optional[List[int]]

    def __init__(self, shape: ShapeType, 
                 dtype: Type = torch.float32, 
                 max_values: Optional[List[int]] = None):
        super().__init__(torch.Tensor)
        if max_values is not None and isinstance(shape[-1], int):
            assert len(max_values) == shape[-1]
        self.shape = shape
        self.dtype = dtype
        self.max_values = max_values

    def get_params_repr(self) -> str:
        ret = []
        for item in self.shape:
            if isinstance(item, ShapeVar):
                s = item.get_name()
            else:
                s = repr(item)
            ret.append(s)
        ret = "(" + ", ".join(ret) + f")"
        # only print dtype if it ain't float
        if self.dtype != torch.float32:
            ret += f", dtype={str(self.dtype).split('.')[-1]}"
        if self.max_values is not None:
            ret += f", max_values={self.max_values}"
        return ret

    def __len__(self):
        return 1

    def __getitem__(self, idx) -> "TensorTD":
        if not isinstance(idx, tuple):
            idx = (idx,)
        ret_shape = []
        ret_max_values = self.max_values
        for axis, (i, sz) in enumerate(zip(idx, self.shape)):
            if axis == len(self.shape) - 1 and self.max_values is not None:
                ret_max_values = self.max_values[i]
                if isinstance(i, int):
                    ret_max_values = [ ret_max_values ]

            if isinstance(i, int):
                continue
            if isinstance(i, slice):
                if i.step is not None:
                    raise NotImplementedError()
                start = 0 if i.start is None else normalize_idx(sz, i.start)
                stop = sz if i.stop is None else normalize_idx(sz, i.stop)
                if isinstance(start, int) and isinstance(stop, int):
                    assert stop >= start
                    assert stop <= sz
                ret_shape.append(stop-start)
        ret_shape += list(self.shape[len(idx):])
        ret = deepcopy(self)
        ret.shape = tuple(ret_shape)
        ret.max_values = ret_max_values
        return ret

    @property
    def max_value(self):
        assert self.max_values is not None and len(self.max_values) == 1
        return self.max_values[0]

class ClassTD(TypeData):
    """ Type data for custom data types """

    subtypes: Dict[str, TypeData]

    def __init__(self, runtime_type, **kwargs):
        super().__init__(runtime_type)
        self.subtypes = {}
        for key, val in kwargs.items():
            self.subtypes[key] = val

    def __getattr__(self, key: str) -> TypeData:
        if key in [ "runtime_type", "subtypes" ]:
            raise AttributeError
        else:
            try:
                return self.subtypes[key]
            except KeyError:
                raise AttributeError
        
    def __setattr__(self, key: str, val: TypeData):
        if key in [ "runtime_type", "subtypes" ]:
            super().__setattr__(key, val)
        else:
            self.subtypes[key] = val

    def get_params_repr(self) -> str:
        ret = []
        for key, val in self.subtypes.items():
            ret.append(f"{key}={val.get_name()}")
        return ", ".join(ret)
