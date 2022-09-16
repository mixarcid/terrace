from typing import TypeVar, Generic, Type, Any, Dict, List, Union

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from module import Module
from type_data import ClassTD
from meta_utils import classclass, default_init

@default_init
class Batchable:
    """ Extremely hacky. Need to allow for dict and list inputs to
    modules. But proof of concept """

    @classclass
    class Module(Module):
        many_inputs = True
        
        def __init__(self, batch=True, **kwargs):
            prev_modules = list(kwargs.values())
            super().__init__(prev_modules, batch, kwargs.keys())
            self.batch = batch

        def get_out_types(self, batch, keys):
            self.keys = keys
            subclass = type(self).Owner
            typ = Batch[subclass] if batch else subclass
            return [ ClassTD(typ, **{ key: mod.out_type for key, mod in zip(keys, self.prev_modules) }) ]

        def __call__(self, *inputs):
            subclass = type(self).Owner
            kwargs = { key: inp for key, inp in zip(self.keys, inputs) }
            if self.batch:
                return Batch(subclass, **kwargs)
            else:
                return subclass(**kwargs)

T = TypeVar('T')

class Batch(Generic[T]):

    batch_type: Type
    batch_subtypes: Dict[str, Type]
    batch_size: int
    store: Dict[str, Any]

    def __init__(self,
                 items: Union[List[T], Type],
                 **kwargs):

        if isinstance(items, type):
            self.init_from_type(items, **kwargs)
        else:
            self.init_from_list(items)

    def init_from_list(self, items: List[T]):
        assert len(items) > 0
        template = items[0]

        self.batch_type = type(template)
        self.batch_subtypes = {}
        self.batch_size = len(items)
        self.store = {}
        
        for key, val in template.__dict__.items():
            self.batch_subtypes[key] = type(val)
            attribs = [ getattr(item, key) for item in items ]
            self.store[key] = collate(attribs)

    def init_from_type(self, batch_type: Type, **kwargs):
        self.batch_type = batch_type
        self.batch_subtypes = batch_type.__annotations__
        template_key = next(iter(self.batch_subtypes.keys()))
        self.batch_size = len(kwargs[template_key])
        self.store = {}

        for key in self.batch_subtypes.keys():
            val = kwargs[key]
            assert len(val) == self.batch_size
            setattr(self, key, val)
            

    def __len__(self) -> int:
        return self.batch_size

    def __getattr__(self, key: str) -> Any:
        try:
            return self.store[key]
        except KeyError:
            raise AttributeError
        
    def __setattr__(self, key: str, val: Any):
        if key in [ 'batch_type', 'batch_subtypes', 'batch_size', 'store' ]:
            super().__setattr__(key, val)
        else:
            self.store[key] = val

    def __getitem__(self, index: int) -> T:
        if index >= self.batch_size:
            raise IndexError
        ret = self.batch_type.__new__(self.batch_type)
        for key, val in self.store.items():
            item = val[index]
            item_type = self.batch_subtypes[key]
            if not isinstance(item, item_type):
                item = item_type(item)
            setattr(ret, key, item)
        return ret
            
def collate(batch: Any) -> Any:
    example = batch[0]
    if isinstance(example, Batchable):
        return Batch(batch)
    else:
        return default_collate(batch)

class DataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=collate, **kwargs
        )
