from typing import Dict, Tuple
from src.terrace import Batch, Batchable, collate, DataLoader
import torch

def test_basic_batching():
    
    class Test(Batchable):
        
        test1: torch.Tensor
        test2: torch.Tensor

    item = Test(torch.zeros((10,10)), torch.zeros((20, 5)))
    batch = Batch([item, item, item])

    assert batch.test1.shape == (3, 10, 10)
    assert batch.test2.shape == (3, 20, 5)

    assert len(batch) == 3

    for item2 in batch:
        assert item2.test1.shape == (10, 10)
        assert item2.test2.shape == (20, 5)

def test_sub_batching():

    class SubTest(Batchable):
        
        test1: torch.Tensor
        test2: torch.Tensor

    class Test(Batchable):
        sub_test: SubTest
        test3: torch.Tensor

    item = Test(SubTest(torch.zeros((10,10)), torch.zeros((20, 5))), torch.zeros(15,))

    batch = Batch([item, item, item])

    assert isinstance(batch.test3, torch.Tensor)
    assert isinstance(batch.sub_test, Batch)

    assert batch.sub_test.test1.shape == (3, 10, 10)
    assert batch.sub_test.test2.shape == (3, 20, 5)
    assert batch.test3.shape == (3, 15)

    for item2 in batch:
        assert item2.sub_test.test1.shape == (10, 10)
        assert item2.sub_test.test2.shape == (20, 5)
        assert item2.test3.shape == (15,)

def test_collate():

    class SubTest(Batchable):
        
        test1: torch.Tensor
        test2: torch.Tensor

    class Test(Batchable):
        sub_test: SubTest
        test3: torch.Tensor

    item = Test(SubTest(torch.zeros((10,10)), torch.zeros((20, 5))), torch.zeros(15,))
    to_collate = (item, "aaa", 3)
    batch = collate([to_collate, to_collate])
    print(batch)

    assert len(batch) == 3
    item_batch, str_batch, int_batch = batch
    
    assert isinstance(item_batch, Batch)
    assert len(item_batch) == 2
    assert item_batch.sub_test.test1.shape == (2, 10, 10)
    assert item_batch.sub_test.test2.shape == (2, 20, 5)
    assert item_batch.test3.shape == (2, 15)

    assert isinstance(int_batch, torch.Tensor)
    assert int_batch.shape == (2,)
    assert int_batch.dtype == torch.long

    assert isinstance(str_batch, list)
    assert len(str_batch) == 2
    assert str_batch[-1] == "aaa"

def test_obj_collate():

    class Obj:
        
        def __init__(self):
            self.test = 1

    batch = collate([Obj(), Obj(), Obj()])
    assert isinstance(batch, list)
    assert isinstance(batch[0], Obj)
    assert batch[0].test == 1

def test_dataloader():

    class SubTest(Batchable):
        
        test1: torch.Tensor
        test2: torch.Tensor

    class Test(Batchable):
        sub_test: SubTest
        test3: torch.Tensor

    class TestDataset(torch.utils.data.Dataset):

        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index > len(self): raise IndexError()
            item = Test(SubTest(torch.zeros((10,10)), torch.zeros((20, 5))), torch.zeros(15,))
            return {
                "item": item,
                "more": (item, 4.0)
            }

    loader = DataLoader(TestDataset(), batch_size=2)
    batch = next(iter(loader))
    
    assert isinstance(batch, dict)
    item_batch = batch["item"]
    
    assert isinstance(item_batch, Batch)
    assert len(item_batch) == 2
    assert item_batch.sub_test.test1.shape == (2, 10, 10)
    assert item_batch.sub_test.test2.shape == (2, 20, 5)
    assert item_batch.test3.shape == (2, 15)

    item_batch, float_batch = batch["more"]

    assert isinstance(item_batch, Batch)
    assert len(item_batch) == 2
    assert item_batch.sub_test.test1.shape == (2, 10, 10)
    assert item_batch.sub_test.test2.shape == (2, 20, 5)
    assert item_batch.test3.shape == (2, 15)

    assert isinstance(float_batch, torch.Tensor)
    assert float_batch.shape == (2,)
    assert float_batch.dtype == torch.float64

def test_complex_batchable():

    class SubTest(Batchable):
        
        test1: Tuple[int, float]
        test2: Dict[str, torch.Tensor]

    class Test(Batchable):
        sub_test: SubTest

    item = Test(SubTest((1, 2.9), {"a": torch.zeros(2,3), "b": torch.zeros(1,)}))
    batch = collate([item, item, item])

    assert isinstance(batch.sub_test.test1, tuple)
    int_batch, float_batch = batch.sub_test.test1

    assert isinstance(int_batch, torch.Tensor)
    assert int_batch.shape == (3,)
    assert int_batch.dtype == torch.long

    assert isinstance(float_batch, torch.Tensor)
    assert float_batch.shape == (3,)
    assert float_batch.dtype == torch.float64

    assert isinstance(batch.sub_test.test2, dict)
    a_batch = batch.sub_test.test2["a"]
    b_batch = batch.sub_test.test2["b"]

    assert a_batch.shape == (3,2,3)
    assert b_batch.shape == (3,1)

def test_make_batch():

    class SubTest(Batchable):
        
        test1: Tuple[int, float]
        test2: Dict[str, torch.Tensor]

    class Test(Batchable):
        sub_test: SubTest
        test3: torch.Tensor

    sub_batch = Batch(SubTest, test1=(torch.zeros((2,), dtype=torch.long), torch.ones((2,), dtype=torch.float32)), test2={"a": torch.zeros(2,3)})
    batch = Batch(Test, sub_test=sub_batch, test3=torch.zeros(2,5,10))

    for item in batch:
        
        int_item = item.sub_test.test1[0].item()
        float_item = item.sub_test.test1[1].item()

        assert isinstance(int_item, int)
        assert int_item == 0

        assert isinstance(float_item, float)
        assert float_item == 1.0

        assert item.sub_test.test2["a"].shape == (3,)

        assert item.test3.shape == (5,10)

def test_make_batch_exc():
    raise NotImplementedError()

def test_custom_collate():
    raise NotImplementedError()

def test_batch_view_batch():

    class SubTest(Batchable):
        
        test1: Tuple[int, float]
        test2: Dict[str, torch.Tensor]

    class Test(Batchable):
        sub_test: SubTest

    item = Test(SubTest((1, 2.9), {"a": torch.zeros(2,3), "b": torch.zeros(1,)}))
    batch = collate([item, item, item])

    batch = collate([batch[0], batch[2]])

    assert isinstance(batch.sub_test.test1, tuple)
    int_batch, float_batch = batch.sub_test.test1

    assert isinstance(int_batch, torch.Tensor)
    assert int_batch.shape == (2,)
    assert int_batch.dtype == torch.long

    assert isinstance(float_batch, torch.Tensor)
    assert float_batch.shape == (2,)
    assert float_batch.dtype == torch.float64

    assert isinstance(batch.sub_test.test2, dict)
    a_batch = batch.sub_test.test2["a"]
    b_batch = batch.sub_test.test2["b"]

    assert a_batch.shape == (2,2,3)
    assert b_batch.shape == (2,1)


def test_batch_slicing():
    raise NotImplementedError()

def test_batch_cuda():
    
    class SubTest(Batchable):
        
        test1: Tuple[int, float]
        test2: Dict[str, torch.Tensor]

    class Test(Batchable):
        sub_test: SubTest

    item = Test(SubTest((1, 2.9), {"a": torch.zeros(2,3), "b": torch.zeros(1,)}))
    batch = collate([item, item, item])

    batch_cu = batch.cuda()
    assert batch_cu.sub_test.test2["a"].device.type == "cuda"

    batch_cu = batch.to("cuda")
    assert batch_cu.sub_test.test2["a"].device.type == "cuda"

    batch_cpu = batch_cu.cpu()
    assert batch_cpu.sub_test.test2["a"].device.type == "cpu"

def test_custom_batch_method(self):
    class Test(Batchable):

        def batch_test(self):
            return self.t1

        t1: torch.Tensor

    test = Test(t1=torch.randn(1,3))
    batch = collate([test, test, test])
    assert torch.all(batch.test() == batch.t1)