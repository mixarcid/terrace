from typing import List, Set, Tuple, Optional, Type, TypeVar, Generic, Any
import dgl
import torch

from batch import Batchable, Batch, TypeTree, make_batch

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
            new_edata.append(edata[i])
            if not directed:
                src_list.append(n2)
                dst_list.append(n1)
                new_edata.append(edata[i])
        
        self.dgl_graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=len(nodes), idtype=torch.int32)

        node_batch = Batch(nodes)
        self.node_type_tree = node_batch.type_tree
        for key, val in node_batch.store.items():
            self.dgl_graph.ndata[key] = val

        if new_edata is not None:
            edge_batch = Batch(new_edata)
            self.edge_type_tree = edge_batch.type_tree
            for key, val in edge_batch.store.items():
                self.dgl_graph.edata[key] = val
        else:
            self.edge_type_tree = None

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
        ret.batch_size = self.dgl_graph.number_of_nodes()
        ret.type_tree = self.edge_type_tree
        ret.store = self.dgl_graph.edata
        return ret


class GraphBatch(Batch[Graph[N, E]]):

    dgl_batch: dgl.batch
    node_type_tree: TypeTree
    edge_type_tree: Optional[TypeTree]
    
    def __init__(self, items: List[Graph[N, E]]):
        assert len(items) > 0
        first = items[0]
        self.node_type_tree = first.node_type_tree
        self.edge_type_tree = first.edge_type_tree
        self.dgl_batch = dgl.batch([ item.dgl_graph for item in items ])

    def __len__(self) -> int:
        return self.dgl_batch.batch_size

    def __getitem__(self, index: int) -> Graph[N, E]:
        if index >= len(self):
            raise IndexError
        ret = Graph.__new__(Graph)
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
        ret.batch_size = self.dgl_batch.number_of_nodes()
        ret.type_tree = self.edge_type_tree
        ret.store = self.dgl_batch.edata
        return ret
    
if __name__ == "__main__":

    class SubNTest(Batchable):
        t1: torch.Tensor

    class NTest(Batchable):
        t1: SubNTest
        t2: torch.Tensor
    
    class Edata(Batchable):
        et1: torch.Tensor
    
    
    nodes = [ NTest(SubNTest(torch.tensor([1,0,0])), torch.tensor([0,1,0])) for n in range(10) ]
    edges = [(0,1), (1,2)]
    edata = [ Edata(torch.tensor([0,0,1])) for e in edges ]
    graph = Graph(nodes, edges, edata)
    print(graph.ndata.t1.t1)
    graph.dgl_graph
    
    batch = make_batch([graph, graph])
    for g in batch:
        print(g.edata.et1)
