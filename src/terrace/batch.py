from typing import Any, Generic, Optional, Type, TypeVar, Union, List, Tuple
from dataclassy import dataclass
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from .categorical_tensor import CategoricalTensor
from .meta_utils import recursive_map


@dataclass
class Batchable:
    """ Base class for all objects we want to batchify
    
    This method can also define static methods collate_{attribute}
    and index_{attribute} """
    
    @staticmethod
    def get_batch_type():
        """ override this method if you want a custom batch type
        for your batchable class """
        return Batch

    def asdict(self):
        """ Override this method to define what attributes you want your
        Batch to define """
        return self.__dict__

T = TypeVar('T')

class BatchBase(Generic[T]):

    def item_type(self) -> Type[Batchable]:
        """ Returns the type of each item in the batch """
        raise NotImplementedError()

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

class Batch(BatchBase[T]):

    _internal_attribs = [ "_batch_len", "_batch_type" ]
    _batch_len: int
    _batch_type: Type[Batchable]

    def __init__(self,
                items: Optional[Union[List[T], Type[T]]],
                **kwargs):
        if isinstance(items, type):
            self._init_from_type(items, **kwargs)
        else:
            self._init_from_list(items)

    def _init_from_list(self, items: List[T]):

        assert len(items) > 0
        template = items[0]
        template_type = type(template)

        self._batch_len = len(items)
        self._batch_type = template_type

        if isinstance(template, BatchBase):
            raise ValueError("Batchifying batches is not supported at the moment")

        attribs = { key: [] for key in template.asdict().keys() }
        for item in items:
            for key, attrib in item.asdict().items():
                attribs[key].append(attrib)

        for key, attrib_list in attribs.items():

            if key in Batch._internal_attribs:
                raise ValueError(f"{key} is used internally by Batch, so it shouldn't be a member of {template_type}")

            collate_method = "collate_" + key
            if hasattr(type(template), collate_method):
                collated = getattr(template_type, collate_method)(attrib_list)
            else:
                collated = collate(attrib_list)

            self.__dict__[key] = collated
        

    def _init_from_type(self, batch_type: Type, **kwargs):
        self._batch_type = batch_type
        if len(kwargs) == 0:
            raise ValueError("Can't determine batch size of empty batch")
        self._batch_len = len(next(iter(kwargs.values())))
        for key, val in kwargs.items():
            self.__dict__[key] = val

    def __len__(self) -> int:
        return self._batch_len

    def __getitem__(self, index) -> "BatchView[T]":
        if isinstance(index, int) and index >= len(self):
            raise IndexError()
        return BatchView(self, index)

    def item_type(self) -> Type[T]:
        """ Returns the type of each item in the batch """
        return self._batch_type

    def attribute_names(self) -> List[str]:
        """ Returns the names of all the batched attributes """
        return [ key for key in self.__dict__.keys() if key not in Batch._internal_attribs ]

    def asdict(self):
        """ Convert to dict """
        return { key: val for key, val in self.__dict__.items() if key not in Batch._internal_attribs }

    def to(self, device):
        to_dict = { key: recursive_map(lambda x: x.to(device) if hasattr(x, 'to') else x, val) for key, val in self.asdict().items() }
        return Batch(self._batch_type, **to_dict)

class BatchView(Generic[T], Batchable):
    """ View of an item in a batch. Should act like said item in most
    circumstances. We use views instead of creating actual items because,
    for many use cases, lazily indexing batches is much faster """

    _internal_attribs = [ "_batch", "_index" ]
    _batch: Batch[T]
    _index: Union[int, slice] # either int or slice

    def __init__(self, batch: Batch, index: Union[int, slice]):
        self._batch = batch
        self._index = index

    def __getattribute__(self, name: str) -> Any:
        if name in BatchView._internal_attribs:
            return object.__getattribute__(self, name)
        if name in self._batch.__dict__:
            attrib = getattr(self._batch, name)

            item_type = self._batch.item_type()
            index_method = "index_" + name
            if hasattr(item_type, index_method):
                return getattr(item_type, index_method)(attrib, self._index)

            if isinstance(attrib, tuple):
                return tuple([ item[self._index] for item in attrib ])
            elif isinstance(attrib, dict):
                return { key: val[self._index] for key, val in attrib.items() }
            return attrib[self._index]
        return object.__getattribute__(self, name)

    def asdict(self):
        return { key: getattr(self, key) for key in self._batch.attribute_names() }

def collate(batch: Any) -> Any:
    """ turn a list of items into a batch of items. Replacement
    for pytorch's default collate. This is what we use in the
    custom DataLoader class """
    # performance optimization -- if we've already batched something, no
    # need to do it again
    if isinstance(batch, Batch):
        return batch

    example = batch[0]
    if isinstance(example, Batchable):
        return type(example).get_batch_type()(batch)
    elif isinstance(example, tuple) or isinstance(example, list):
        ret = []
        for i, item in enumerate(example):
            all_items = [ b[i] for b in batch]
            ret.append(collate(all_items))
        return type(example)(ret)
    elif isinstance(example, dict):
        ret = {}
        for key in example.keys():
            to_collate = []
            for item in batch:
                to_collate.append(item[key])
            ret[key] = collate(to_collate)
        return ret
    elif isinstance(example, CategoricalTensor):
        return torch.stack(batch)
    else:
        try:
            return default_collate(batch)
        except TypeError:
            return batch

class DataLoader(torch.utils.data.DataLoader):
    """ Dataloader that correctly batchifies Batchable data """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=collate, **kwargs
        )