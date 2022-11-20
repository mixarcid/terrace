from typing import TypeVar, Generic, Type, Any, Dict, List, Union, Sequence, Optional

from copy import copy
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from dataclasses import dataclass

from .type_data import TypeData, ClassTD, TensorTD, ShapeVar
from .meta_utils import classclass, get_type_name, default_init, recursive_map

class Batchable:

    def __init__(self, *args, **kwargs):
        """ default constructor that gets inherited """
        # todo: add error messages
        seen = set()
        assert len(args) <= len(self.__annotations__)
        for name, arg in zip(self.__annotations__.keys(), args):
            setattr(self, name, arg)
            seen.add(name)
            
        for key, val in kwargs.items():
            assert key not in seen
            assert key in self.__annotations__
            setattr(self, key, val)
            
    @staticmethod
    def get_batch_type():
        """ override this method if you want a custom batch type
        for your batchable class """
        raise NotImplementedError()

    @staticmethod
    def get_batch_td_type():
        """ If you're using a custom batch type, you'll also need a custom
        BatchTD type """
        raise NotImplementedError()

@dataclass
class TypeTree:
    type: Type
    subtypes: Dict[str, "TypeTree"]
            
T = TypeVar('T')

class BatchBase(Generic[T]):
    pass

class Batch(BatchBase[T]):

    type_tree: TypeTree
    batch_size: int
    store: Dict[str, Any]

    def __init__(self,
                 items: Optional[Union[List[T], Type]],
                 **kwargs):
        # items should only be None when we're making a blank batch
        # and want to manually mess with things (e.g. in get and set attr)
        if items is None: return
        if isinstance(items, type):
            self.init_from_type(items, **kwargs)
        else:
            self.init_from_list(items)

    def init_from_list(self, items: List[T]):
        assert len(items) > 0
        template = items[0]

        self.type_tree = TypeTree(type(template), {})
        self.batch_size = len(items)
        self.store = {}

        if isinstance(template, Batch):
            # recursion out the wazoo
            raise NotImplementedError
        else:
            for key, val in template.__dict__.items():
                self.type_tree.subtypes[key] = TypeTree(type(val), {})
                attribs = [ getattr(item, key) for item in items ]
                collated = collate(attribs)
                if isinstance(collated, Batch):
                    self.type_tree.subtypes[key] = collated.type_tree
                    for key2, val2 in collated.store.items():
                        full_key = key + "/" + key2
                        self.store[full_key] = val2
                else:
                    self.store[key] = collate(attribs)
                
    def init_from_type(self, batch_type: Type, **kwargs):
        self.type_tree = TypeTree(batch_type, batch_type.__annotations__)
        template_key = next(iter(self.type_tree.subtypes.keys()))
        self.batch_size = len(kwargs[template_key])
        self.store = {}

        for key in self.type_tree.subtypes.keys():
            val = kwargs[key]
            assert len(val) == self.batch_size
            setattr(self, key, val)
            

    def __len__(self) -> int:
        return self.batch_size

    def __getattr__(self, key: str) -> Any:
        if key in [ 'type_tree', 'batch_size', 'store' ]:
            raise AttributeError
        if key in self.store:
            return self.store[key]
        else:
            if key in self.type_tree.subtypes:
                sub_batch = Batch(None)
                sub_batch.type_tree = self.type_tree.subtypes[key]
                sub_batch.batch_size = len(self)
                sub_batch.store = {}
                for key2, val in self.store.items():
                    if key2.startswith(key):
                        first, *rest = key2.split("/")
                        new_key = "/".join(rest)
                        sub_batch.store[new_key] = val
                return sub_batch
            else:
                raise AttributeError
        
    def __setattr__(self, key: str, val: Any):
        # for the sake of the subclasses
        if type(self) != Batch:
            return super().__setattr__(key, val)
        if key in [ 'type_tree', 'batch_size', 'store' ]:
            return super().__setattr__(key, val)
        self.store[key] = val

    def __getitem__(self, index: int) -> T:
        if index >= self.batch_size:
            raise IndexError
        ret = self.type_tree.type.__new__(self.type_tree.type)
        for key in self.type_tree.subtypes:
            item = getattr(self, key)[index]
            item_type = self.type_tree.subtypes[key].type
            # if not isinstance(item, item_type):
            #    item = item_type(item)
            setattr(ret, key, item)
        return ret

    def to(self, device):
        ret = Batch(None)
        ret.type_tree = self.type_tree
        ret.batch_size =  self.batch_size
        ret.store = { key: recursive_map(lambda x: x.to(device), val) for key, val in self.store.items() }
        return ret

@default_init
class BatchIter:
    batch_td: "BatchTD"
    index: int

    def __next__(self):
        if self.index != 0:
            raise StopIteration()
        self.index += 1
        return self.batch_td[0]

class BatchTD(ClassTD):

    def __init__(self, type_data: ClassTD, batch_size: Union[int, ShapeVar]):
        subtypes = {}
        for name, subtype in type_data.subtypes.items():
            subtypes[name] = make_batch_td(subtype, batch_size)
        super(BatchTD, self).__init__(Batch[type_data.runtime_type], **subtypes)

    def __getitem__(self, index):
        runtime_type = self.runtime_type.__args__[0]
        subtypes = {}
        for name, subtype in self.subtypes.items():
            subtypes[name] = subtype[index]
        return ClassTD(runtime_type, **subtypes)

    def __iter__(self):
        return BatchIter(self, 0)

def make_batch_td(type_data: TypeData, 
                  batch_size: Union[int, ShapeVar] = ShapeVar('B')) -> TypeData:
    if isinstance(type_data, tuple):
        return tuple(make_batch_td(item) for item in type_data)
    elif isinstance(type_data, TensorTD):
        ret = copy(type_data)
        ret.shape = tuple([batch_size] + list(type_data.shape))
        return ret
    else:
        try:
            return type_data.runtime_type.get_batch_td_type()(type_data, batch_size)
        except NotImplementedError:
            return BatchTD(type_data, batch_size)

def make_batch(items):
    assert len(items) > 0
    first = items[0]
    try:
        return type(first).get_batch_type()(items)
    except (NotImplementedError, AttributeError):
        return Batch(items)
            
def collate(batch: Any) -> Any:
    example = batch[0]
    if isinstance(example, Batchable):
        return make_batch(batch)
    elif isinstance(example, tuple) or isinstance(example, list):
        ret = []
        for i, item in enumerate(example):
            all_items = [ b[i] for b in batch]
            ret.append(collate(all_items))
        return type(example)(ret)
    elif isinstance(example, dict):
        raise NotImplementedError()
    else:
        return default_collate(batch)

class DataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=collate, **kwargs
        )

    def get_type_data(self) -> TypeData:
        """ If the dataset impliments get_type_data, this will return the
        batchfied type data """
        return make_batch_td(self.dataset.get_type_data())

if __name__ == "__main__":

    class Test(Batchable):
        t1: torch.Tensor
        t2: list

    class Test2(Batchable):
        item1: Test
        item2: Test

    class Test3(Batchable):
        fred: Test2
        george: Test

    test = Test(torch.tensor([1,2,3]), t2=[1,2,3])
    test2 = Test2(test, test)
    test3 = Test3(test2, test)
    batch = Batch([test, test])
    batch3 = Batch([test3, test3, test3])
    print(batch3.george.t1)
    
