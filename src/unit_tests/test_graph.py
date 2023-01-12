from typing import Dict, Tuple
from src.terrace import Batch, Batchable, collate, DataLoader, Graph
from src.terrace.graph import flatten_batch, unflatten_batch
import torch

def test_flatten_batch():

    class SubTest(Batchable):
        
        test1: torch.Tensor
        test2: torch.Tensor

    class Test(Batchable):
        sub_test: SubTest
        test3: torch.Tensor

    item = Test(SubTest(torch.zeros((10,10)), torch.zeros((20, 5))), torch.zeros(15,))

    batch = Batch([item, item, item])
    flat, type_tree = flatten_batch(batch)
    for key, val in flat.items():
        assert isinstance(key, str)
        assert isinstance(val, torch.Tensor)

    batch = unflatten_batch(flat, type_tree)

    assert isinstance(batch.test3, torch.Tensor)
    assert isinstance(batch.sub_test, Batch)

    assert batch.sub_test.test1.shape == (3, 10, 10)
    assert batch.sub_test.test2.shape == (3, 20, 5)
    assert batch.test3.shape == (3, 15)

    for item2 in batch:
        assert item2.sub_test.test1.shape == (10, 10)
        assert item2.sub_test.test2.shape == (20, 5)
        assert item2.test3.shape == (15,)

def test_graph_batch():
    class SubNTest(Batchable):
        t1: torch.Tensor

    class NTest(Batchable):
        t1: SubNTest
        t2: torch.Tensor
    
    class Edata(Batchable):
        et1: torch.Tensor

    class TwoGraphs(Batchable):
        g1: Graph
        g2: Graph
    
    nodes = [ NTest(SubNTest(torch.tensor([1,0,0])), torch.tensor([0,1,0,0])) for n in range(10) ]
    edges = [(0,1), (1,2)]
    edata = [ Edata(torch.tensor([0,0,1])) for e in edges ]
    graph = Graph(nodes, edges, edata)
    
    assert graph.ndata.t1.t1.shape == (10, 3)
    assert graph.ndata.t2.shape == (10, 4)
    assert graph.edata.et1.shape == (4, 3)
    assert len(graph.edges) == 4

    for edge in graph.edges:
        assert edge in [(0,1), (1,2), (1,0), (2,1)]

    graph.dgl()

    batch = collate([graph, graph])
    assert batch.ndata.t1.t1.shape == (20, 3)
    assert batch.ndata.t2.shape == (20, 4)
    assert batch.edata.et1.shape == (8, 3)
    assert len(batch.edges) == 8

    batch.dgl()

    # are all the edges unique?
    assert len(set(batch.edges)) == len(batch.edges)

    for g in batch:
        assert g.ndata.t1.t1.shape == (10, 3)
        assert g.ndata.t2.shape == (10, 4)
        assert g.edata.et1.shape == (4, 3)
        assert len(g.edges) == 4
        
        for edge in g.edges:
            assert edge in [(0,1), (1,2), (1,0), (2,1)]

        g.dgl()

    tg = TwoGraphs(graph, graph)
    two_batch = collate([tg, tg, tg])

    assert two_batch.g1.ndata.t1.t1.shape == (30, 3)
    assert two_batch.g1.ndata.t2.shape == (30, 4)
    assert two_batch.g1.edata.et1.shape == (12, 3)
    assert len(two_batch.g1.edges) == 12

    two_batch.g1.dgl()
    two_batch[0].g1.dgl()

def test_graph_batch_view_batch():
    raise NotImplementedError

def test_graph_batch_slices():
    raise NotImplementedError

def test_graph_cuda():
    class SubNTest(Batchable):
        t1: torch.Tensor

    class NTest(Batchable):
        t1: SubNTest
        t2: torch.Tensor
    
    class Edata(Batchable):
        et1: torch.Tensor

    class TwoGraphs(Batchable):
        g1: Graph
        g2: Graph
    
    nodes = [ NTest(SubNTest(torch.tensor([1,0,0])), torch.tensor([0,1,0,0])) for n in range(10) ]
    edges = [(0,1), (1,2)]
    edata = [ Edata(torch.tensor([0,0,1])) for e in edges ]
    graph = Graph(nodes, edges, edata)

    batch = collate([TwoGraphs(graph, graph)])
    batch_cu = batch.cuda()

    assert batch_cu.g1.dgl().device.type == "cuda"

    for item in batch_cu:
        assert item.g1.dgl().device.type == "cuda"

