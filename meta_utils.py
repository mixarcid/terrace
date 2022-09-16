# utility functions for python metaprogramming

from dataclasses import dataclass
from copy import deepcopy

def default_init(cls):
    """ makes a default init function for a class """
    return dataclass(eq=False,repr=False)(cls)

class classclass():
    """ Allows a class within a class to access subclasses
    of the owner class """
    def __init__(self, cls):
        self.cls = cls
    def __get__(self, instance, cls):
        new_cls = deepcopy(self.cls)
        new_cls.Owner = cls
        return new_cls

def get_type_name(typ):
    """ Returns the name of a type even if it's a generic alias
    (because python is weird and something like List[int] won't have
    a __name__ but List will) """
    if hasattr(typ, "__name__"):
        return typ.__name__
    elif hasattr(typ, "__origin__"):
        return typ.__origin__.__name__ + "[" + ", ".join(map(get_type_name, typ.__args__)) + "]"
