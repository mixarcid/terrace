import torch
from typing import Type, List, Tuple, Union
from dataclasses import dataclass

def default_init(cls):
    """ makes a default init function for a class """
    return dataclass(eq=False,repr=False)(cls)

class ShapeVar:
    """ If we don't know the dimension of something at graph time,
    put it in a shape var (e.g. batch) """

    # for everything in this library, get_name is used when pretty-printing
    # or exporting models, to pdfs, __repr__ is used for regular printing
    def get_name(self) -> str:
        raise NotImplementedError

    def get_inner_name(self) -> str:
        """ Inside math expressions we may want to put parens
        around. This function allows that """
        return self.get_name()

    def __repr__(self) -> str:
        return f"ShapeVar[{self.get_name()}]"

    def __add__(self, other: "ShapeVar"):
        return AddShapeVar([self, other])

    def __sub__(self, other: "ShapeVar"):
        return SubShapeVar([self, other])

    def __mul__(self, other: "ShapeVar"):
        return MulShapeVar([self, other])
    
    def __floordiv__(self, other: "ShapeVar"):
        return DivShapeVar([self, other])

@default_init
class NamedShapeVar(ShapeVar):
    name: str

    def get_name(self) -> str:
        return self.name

@default_init
class OpShapeVar(ShapeVar):

    op_name: str
    ops: List[ShapeVar]

    def get_name(self) -> str:
        return f" {self.op_name} ".join([ s.get_inner_name() for s in self.ops ])

    def get_inner_name(self) -> str:
        return f"({self.get_name()})"
    
class AddShapeVar(OpShapeVar):

    def __init__(self, ops):
        super().__init__("+", ops)

class SubShapeVar(OpShapeVar):

    def __init__(self, ops):
        super().__init__("-", ops)

class MulShapeVar(OpShapeVar):

    def __init__(self, ops):
        super().__init__("*", ops)

class DivShapeVar(OpShapeVar):

    def __init__(self, ops):
        super().__init__("//", ops)

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
