from typing import List, Set, Tuple, Optional, Type, TypeVar, Generic, Any, Union
import dgl
import torch

from .batch import Batchable, Batch, BatchBase, TypeTree, make_batch, make_batch_td, BatchTD
from .type_data import ClassTD, TypeData, ShapeVar

class Batchable(Batchable):
    pass

N = TypeVar('N', bound=Batchable)
E = TypeVar('E', bound=Optional[Batchable])
class Graph(Generic[N, E], Batchable):
    """ Wrapper around dgl graph allowing easier access to data """

    dgl_graph: dgl.graph
    node_type_tree: TypeTree
    edge_type_tree: Optional[TypeTree]

    @staticmethod
    def get_batch_type():
        return GraphBatch

    @staticmethod
    def get_batch_td_type():
        return GraphBatchTD

    def __init__(self, nodes: List[N],
                 edges: List[Tuple[int, int]],
                 edata: Optional[List[E]] = None,
                 directed: bool = False):
        """ If directed is false, both permutations of the edges
        will be added. """

        # just like 
        if nodes is None: return

        src_list = []
        dst_list = []
        new_edata = None if edata is None else []
        for i, (n1, n2) in enumerate(edges):
            src_list.append(n1)
            dst_list.append(n2)
            # if new_edata is not None:
            #     new_edata.append(edata[i])
            if not directed:
                src_list.append(n2)
                dst_list.append(n1)
                if new_edata is not None:
                    new_edata.append(edata[i])
                    new_edata.append(edata[i])
        
        if directed and edata is not None:
            assert new_edata == []
            new_edata = edata
        
        self.dgl_graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=len(nodes), idtype=torch.int32, device='cpu')

        if isinstance(nodes, Batch):
            node_batch = nodes
        else:
            node_batch = Batch(nodes)
       
        self.node_type_tree = node_batch.type_tree
        for key, val in node_batch.store.items():
            self.dgl_graph.ndata[key] = val

        if new_edata is not None:
            if isinstance(new_edata, Batch):
                edge_batch = new_edata
            else:
                edge_batch = Batch(new_edata)
            self.edge_type_tree = edge_batch.type_tree
            for key, val in edge_batch.store.items():
                self.dgl_graph.edata[key] = val
        else:
            self.edge_type_tree = None

        # hacky fix for https://github.com/dmlc/dgl/issues/3802
        self.dgl_graph.create_formats_()

    @property
    def ndata(self) -> Batch[N]:
        ret = Batch(None)
        ret.batch_size = self.dgl_graph.number_of_nodes()
        ret.type_tree = self.node_type_tree
        ret.store = self.dgl_graph.ndata
        return ret

    @property
    def edata(self) -> Batch[E]:
        if self.edge_type_tree is None:
            raise AttributeError
        ret = Batch(None)
        ret.batch_size = self.dgl_graph.number_of_edges()
        ret.type_tree = self.edge_type_tree
        ret.store = self.dgl_graph.edata
        return ret

    @property
    def edges(self) -> List[Tuple[int, int]]:
        ret = []
        for src, dst in zip(*self.dgl_graph.edges()):
            ret.append((int(src), int(dst)))
        return ret

    def to(self, device):
        ret = Graph.__new__(GraphBatch)
        ret.node_type_tree = self.node_type_tree
        ret.edge_type_tree = self.edge_type_tree
        ret.dgl_batch = self.dgl_graph.to(device)
        return ret

G = TypeVar('G', bound=Graph)
class GraphBatch(BatchBase[G]):

    dgl_batch: dgl.batch
    graph_type: Type[Graph]
    node_type_tree: TypeTree
    edge_type_tree: Optional[TypeTree]
    
    def __init__(self, items: List[Graph[N, E]]):
        assert len(items) > 0
        first = items[0]
        self.graph_type = type(first)
        self.node_type_tree = first.node_type_tree
        self.edge_type_tree = first.edge_type_tree
        self.dgl_batch = dgl.batch([ item.dgl_graph for item in items ])

    def __len__(self) -> int:
        return self.dgl_batch.batch_size

    def __getitem__(self, index: int) -> Graph[N, E]:
        if index >= len(self):
            raise IndexError
        ret = self.graph_type.__new__(self.graph_type)
        ret.dgl_graph = dgl.unbatch(self.dgl_batch)[index]
        ret.node_type_tree = self.node_type_tree
        ret.edge_type_tree = self.edge_type_tree
        return ret

    @property
    def ndata(self) -> Batch[N]:
        ret = Batch(None)
        ret.batch_size = self.dgl_batch.number_of_nodes()
        ret.type_tree = self.node_type_tree
        ret.store = self.dgl_batch.ndata
        return ret

    @property
    def edata(self) -> Batch[E]:
        if self.edge_type_tree is None:
            raise AttributeError
        ret = Batch(None)
        ret.batch_size = self.dgl_batch.number_of_edges()
        ret.type_tree = self.edge_type_tree
        ret.store = self.dgl_batch.edata
        return ret

    @property
    def edges(self) -> List[Tuple[int, int]]:
        ret = []
        for src, dst in zip(*self.dgl_batch.edges()):
            ret.append((src, dst))
        return ret

    def to(self, device):
        from multiprocessing import current_process
        ret = GraphBatch.__new__(GraphBatch)
        ret.graph_type = self.graph_type
        ret.node_type_tree = self.node_type_tree
        ret.edge_type_tree = self.edge_type_tree
        ret.dgl_batch = self.dgl_batch.to(device)
        return ret

class GraphTD(ClassTD):

    def __init__(self, runtime_type: Type[Graph],
                 node_td: TypeData,
                 edge_td: TypeData,
                 node_shapevar: Union[ShapeVar, int] = ShapeVar('N'),
                 edge_shapevar: Union[ShapeVar, int] = ShapeVar('E')):
        ndata = BatchTD(node_td, node_shapevar)
        edata = BatchTD(edge_td, node_shapevar)
        super().__init__(runtime_type, ndata=ndata, edata=edata)

class GraphBatchTD(ClassTD):

    batch_size: Union[ShapeVar, int]

    def __init__(self, graph_td, 
                 batch_size: Union[ShapeVar, int] = ShapeVar("B")):
        subtypes = graph_td.subtypes
        runtime_type = GraphBatch[graph_td.runtime_type]
        super().__init__(runtime_type, **subtypes)
        self.batch_size = batch_size

    def __getattr__(self, key: str) -> TypeData:
        if key == "batch_size":
            return self.__dict__[key]
        else:
            return super().__getattr__(key)
        
    def __setattr__(self, key: str, val: TypeData):
        if key == "batch_size":
            self.__dict__[key] = val
        else:
            super().__setattr__(key, val)
    
if __name__ == "__main__":

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
    
    nodes = [ NTest(SubNTest(torch.tensor([1,0,0])), torch.tensor([0,1,0])) for n in range(10) ]
    edges = [(0,1), (1,2)]
    edata = [ Edata(torch.tensor([0,0,1])) for e in edges ]
    graph = Graph(nodes, edges, edata)
    print(graph.ndata.t1.t1)
    graph.dgl_graph
    
    batch = make_batch([graph, graph])
    for g in batch:
        print(g.edata.et1)

    tg = TwoGraphs(graph, graph)
    print(make_batch([tg, tg, tg]).g1.ndata.t2)
