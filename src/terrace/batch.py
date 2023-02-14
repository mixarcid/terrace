from functools import partial
from torch.utils._pytree import tree_map
from typing import Any, Generic, Optional, Type, TypeVar, Union, List, Tuple
from dataclassy import dataclass
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from .categorical_tensor import CategoricalTensor, NoStackCatTensor
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

def _batch_repr(val):
    if isinstance(val, torch.Tensor):
        val_str = f"Tensor(shape={val.shape}, dtype={val.dtype})"
    else:
        val_str = repr(val)
    return val_str

class Batch(BatchBase[T]):

    _internal_attribs = [ "_batch_len", "_batch_type", "__dict__" ]
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
        template_type = template.__class__ # type(template)
        # todo: very hacky
        if isinstance(template, BatchViewBase):
            template_type = template.get_type()

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
            if hasattr(template_type, collate_method):
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

    def __getitem__(self, index) -> Union["BatchView[T]", Any]:
        if isinstance(index, str):
            return self.__dict__[index]
        if isinstance(index, int) and index >= len(self):
            raise IndexError()
        if isinstance(index, int):
            return BatchView(self, index)
        return collate([self[i] for i in range(len(self))[index]])

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

    def __repr__(self):
        args = []
        for key, val in self.asdict().items():
            val_str = _batch_repr(val)
            args.append(f"{key}={val_str}")
        return f"Batch[{self._batch_type.__name__}]({', '.join(args)})"

    def __getattribute__(self, name: str) -> Any:
        if name in Batch._internal_attribs or name in self.__dict__:
            return object.__getattribute__(self, name)
        batch_method_name = "batch_" + name
        if hasattr(self, "_batch_type") and hasattr(self._batch_type, batch_method_name):
            batch_method = getattr(self._batch_type, batch_method_name)
            if callable(batch_method):
                return partial(batch_method, self)
        return object.__getattribute__(self, name)

class BatchViewBase(Generic[T], Batchable):
    pass

def _get_methods_for_type(type_):
    return {func: getattr(type_, func) for func in dir(type_) if callable(getattr(type_, func)) and not func.startswith("__")}

class BatchView(BatchViewBase[T]):
    """ View of an item in a batch. Should act like said item in most
    circumstances. We use views instead of creating actual items because,
    for many use cases, lazily indexing batches is much faster """

    _internal_attribs = [ "_batch", "_index", "_get_methods" ]
    _batch: Batch[T]
    _index: Union[int, slice] # either int or slice

    def __init__(self, batch: Batch, index: Union[int, slice]):
        self._batch = batch
        self._index = index
        # self.__class__ = self._batch._batch_type

    def _get_methods(self):
        type_ = self._batch._batch_type
        return { key: val for key, val in _get_methods_for_type(type_).items() if key not in _get_methods_for_type(BatchView) }

    def __getattribute__(self, name: str) -> Any:
        if name == "__dict__" or name in BatchView._internal_attribs or name in _get_methods_for_type(BatchView):
            return object.__getattribute__(self, name)
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        if name in self._batch.__dict__ and not name in Batch._internal_attribs:
            attrib = getattr(self._batch, name)

            item_type = self._batch.item_type()
            index_method = "index_" + name
            if hasattr(item_type, index_method):
                ret = getattr(item_type, index_method)(attrib, self._index)
                self.__dict__[name] = ret
                return ret

            if isinstance(attrib, tuple):
                ret = tuple([ item[self._index] for item in attrib ])
                self.__dict__[name] = ret
                return ret
            elif isinstance(attrib, dict):
                ret = { key: val[self._index] for key, val in attrib.items() }
                self.__dict__[name] = ret
                return ret

            ret = attrib[self._index]
            self.__dict__[name] = ret
            return ret

        if isinstance(self._index, int):
            # assume we are acting like a T
            methods = self._get_methods()
            if name in methods:
                return partial(methods[name], self)
        else:
            # assume we are acting like a Batch[T]
            # todo: make this whole "what am I " process more robust
            batch_method_name = "batch_" + name
            if hasattr(self, "_batch") and hasattr(self.get_type(), batch_method_name):
                batch_method = getattr(self.get_type(), batch_method_name)
                if callable(batch_method):
                    return partial(batch_method, self)
            pass
        return object.__getattribute__(self, name)

    def __getitem__(self, index):
        if isinstance(self._index, int):
            raise ValueError("Can only index into a BatchView if the batchview is view of a slice of a batch")
        new_idx = range(len(self._batch))[index]
        if isinstance(new_idx, range):
            new_idx = slice(new_idx.start, new_idx.stop, new_idx.step)
            raise NotImplementedError # need to test this
        else:
            assert isinstance(new_idx, int)
        return BatchView(self._batch, new_idx)

    def asdict(self):
        return { key: getattr(self, key) for key in self._batch.attribute_names() }

    def __repr__(self):
        args = []
        for key, val in self.asdict().items():
            val_str = _batch_repr(val)
            args.append(f"{key}={val_str}")
        return f"BatchView[{self.get_type().__name__}]({', '.join(args)})"

    def get_type(self):
        return self._batch._batch_type

class NoStackTensor(torch.Tensor):
    """ This is used when you want to collate tensors into a list instead
    of stack them. E.g. when you have different shapes"""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        return torch.Tensor._make_subclass(cls, tensor.to('meta'))

    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        # return torch.Tensor.__torch_dispatch__(func, types, args, kwargs)
        def unwrap(x):
            return x.tensor if isinstance(x, NoStackTensor) else x
        def wrap(x):
            return NoStackTensor(x) if isinstance(x, torch.Tensor) else x
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        out = func(*args, **kwargs)
        return tree_map(wrap, out)

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
    elif isinstance(example, NoStackTensor) or isinstance(example, NoStackCatTensor):
        return batch
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