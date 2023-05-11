from typing import Union
from .batch import Batch, Batchable, _batch_repr


class DFRow(Batchable):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val

    def __repr__(self):
        args = []
        for key, val in self.asdict().items():
            val_str = _batch_repr(val)
            args.append(f"{key}={val_str}")
        return f"DFRow({', '.join(args)})"

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

def merge(items: Union[DFRow, Batch]):
    template = items[0]
    attribs = {}
    for item in items:
        attribs.update(item.asdict())
    if isinstance(template, Batch):
        return Batch(DFRow, **attribs)
    else:
        return DFRow(**attribs)