# utility functions for python metaprogramming
from typing import Any, Type, Callable
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


def get_any_arg(*args, **kwargs):
    """ Returns the first argument given, regardless of kwarg or not """
    if len(args) > 0:
        return args[0]
    if len(kwargs) > 0:
        return next(iter(kwargs.values))
    raise AssertionError

def contains_type(item: Any, T: Type):
    """ Returns true if item is an instance of T or
    is a container that contains T """
    if isinstance(item, T):
        return True
    elif isinstance(item, list) or isinstance(item, tuple):
        for i2 in item:
            if contains_type(i2, T):
                return True
    elif isinstance(item, dict):
        for i2 in item.values():
            if contains_type(i2, T):
                return True
    return False

def recursive_map(func: Callable, item: Any):
    """ recurisvely applies func to all non-container
    items in the container """
    
    if isinstance(item, list) or isinstance(item, tuple):
        return type(item)([ recursive_map(func, i2) for i2 in item ])
    elif isinstance(item, dict):
        return { key: recursive_map(func, val) for key, val in item.items() }
    else:
        return func(item)

def recursive_zip(c1, c2):
    """ c1 and c2 are arbitrary containers. They must have the same
    "same" e.g. lists have the same len, dicts have the same keys etc.
    this yeilds the zipped leaves of c1 and c2"""
    
    if isinstance(c1, list) or isinstance(c1, tuple):
        for i1, i2 in zip(c1, c2):
            for tup in recursive_zip(i1, i2): yield tup
    elif isinstance(c1, dict):
        for key in c1:
            for tup in recursive_zip(c1[key], c2[key]): yield tup
    else:
        yield c1, c2