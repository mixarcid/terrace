import torch
from typing import Type, List, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
from sympy import Symbol, Expr, Integer, symbols

def default_init(cls):
    """ makes a default init function for a class """
    return dataclass(eq=False,repr=False)(cls)

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
        try:
            return f"{self.runtime_type.__name__}[{self.get_params_repr()}]"
        except NotImplementedError:
            return self.runtime_type.__name__

    def get_params_repr(self) -> str:
        raise NotImplementedError

ShapeType = Tuple[Union[int, ShapeVar], ...]
        
class TensorTD(TypeData):
    """ Type data for tensor. This is what will be mostly used """
    shape: ShapeType

    def __init__(self, shape: ShapeType):
        super().__init__(torch.Tensor)
        self.shape = shape

    def get_params_repr(self) -> str:
        ret = []
        for item in self.shape:
            if isinstance(item, ShapeVar):
                s = item.get_name()
            else:
                s = repr(item)
            ret.append(s)
        return "(" + ", ".join(ret) + ")"
