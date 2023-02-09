from typing import Dict, List, Sequence, Set, Tuple, Optional, Type, TypeVar, Generic, Any, Union
from dataclassy import dataclass
import dgl
import torch

from .batch import BatchView, BatchViewBase, Batchable, Batch, BatchBase, collate

@dataclass
class TypeTree:
    type: Type
    subtypes: Dict[str, "TypeTree"]

def flatten_batch(b: Batch) -> Tuple[Dict[str, torch.Tensor], TypeTree]:
    """ Flattens batch to a single dict from keys to tensors. Also returns
    all the batch types. This is neccessary because dgl stores ndata and edata
    as such dicts. """

    type_tree = TypeTree(b.item_type(), {})
    ret = {}

    for key, val in b.asdict().items():
        if isinstance(val, torch.Tensor):
            ret[key] = val
            type_tree.subtypes[key] = TypeTree(torch.Tensor, {})
        elif isinstance(val, Batch):
            sub_dict, subtype_tree = flatten_batch(val)
            type_tree.subtypes[key] = subtype_tree
            for key2, val2 in sub_dict.items():
                ret[f"{key}/{key2}"] = val2
        else:
            raise NotImplementedError(f"flatten_batch is only currently implemented for torch.Tensor and Batches, but found attribute {key} with type {type(val)}")

    return ret, type_tree

def unflatten_dict(flat):
    """ Helper function for unflatten_batch. Returns an unflattened dict
    for a dict with keys like 'key/subkey' """
    ret = {}
    for key, val in flat.items():
        container = ret
        subkeys = key.split("/")
        for subkey in subkeys[:-1]:
            if subkey not in container:
                container[subkey] = {}
            container = container[subkey]
        if val == 'None':
            val = None
        container[subkeys[-1]] = val
    return ret

def make_batch_from_unflat_dict(unflat_dict, type_tree):
    """ Helper function for unflatten_batch. Makes a batch from
    a dict that has been unflattened """
    batch_type = type_tree.type
    kwargs = {}
    for key, val in unflat_dict.items():
        if isinstance(val, dict):
            arg = make_batch_from_unflat_dict(val, type_tree.subtypes[key])
        elif isinstance(val, torch.Tensor):
            arg = val
        else:
            raise AssertionError(f"Expected only dicts and tensors, but got {key}={val}")
        kwargs[key] = arg
    return Batch(batch_type, **kwargs)

def unflatten_batch(flat_dict: Dict[str, torch.Tensor], type_tree: TypeTree) -> Batch:
    """ Reverses flatten_batch; takes a flattened dict and a type tree and
    returns a proper Batch """
    unflat_dict = unflatten_dict(flat_dict)
    return make_batch_from_unflat_dict(unflat_dict, type_tree)

N = TypeVar('N', bound=Batchable)
E = TypeVar('E', bound=Optional[Batchable])

@dataclass
class GraphBase(Generic[N, E]):

    _dgl_graph: Union[dgl.graph, dgl.batch]
    _node_type_tree: TypeTree
    _edge_type_tree: Optional[TypeTree]

    @property
    def ndata(self) -> Batch[N]:
        return unflatten_batch(self._dgl_graph.ndata, self._node_type_tree)

    @property
    def edata(self) -> Batch[E]:
        return unflatten_batch(self._dgl_graph.edata, self._edge_type_tree)

    @property
    def edges(self) -> List[Tuple[int, int]]:
        ret = []
        for src, dst in zip(*self._dgl_graph.edges()):
            ret.append((int(src), int(dst)))
        return ret

    def dgl(self) -> Union[dgl.graph, dgl.batch]:
        return self._dgl_graph

    def to(self, device):
        cls_ = type(self)
        ret = cls_.__new__(cls_)
        for key, val in self.__dict__.items():
            if key == "_dgl_graph":
                val = val.to(device)
            ret.__dict__[key] = val
        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ndata={self.ndata}, edata={self.edata})"


class Graph(GraphBase[N, E], Batchable):
    """ Wrapper around dgl graph allowing easier access to data """

    @staticmethod
    def get_batch_type():
        return GraphBatch

    def __init__(self, nodes: Sequence[N],
                 edges: Sequence[Tuple[int, int]],
                 edata: Optional[List[E]] = None,
                 directed: bool = False,
                 device = 'cpu'):
        """ If directed is false, both permutations of the edges
        will be added. """

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
        
        dgl_graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=len(nodes), idtype=torch.int32, device=device)

        node_batch = collate(nodes)
        node_flat_dict, node_type_tree = flatten_batch(node_batch)
        for key, val in node_flat_dict.items():
            dgl_graph.ndata[key] = val

        if new_edata is not None:
            edge_batch = collate(new_edata)
            edge_flat_dict, edge_type_tree = flatten_batch(edge_batch)
            for key, val in edge_flat_dict.items():
                dgl_graph.edata[key] = val
        else:
            edge_type_tree = None

        super().__init__(dgl_graph, node_type_tree, edge_type_tree)

G = TypeVar('G', bound=Graph)
class GraphBatch(BatchBase[G], GraphBase):

    _graph_type: Type[Graph]
    node_slices: List[slice]
    edge_slices: List[slice]
    
    def __init__(self, items: List[Graph[N, E]]):
        dgl_batch = dgl.batch([ item.dgl() for item in items ])
        first = items[0]
        super().__init__(dgl_batch, first._node_type_tree, first._edge_type_tree)
        self._graph_type = type(first)

        # node and edge slices determine which nodes/edges belong to which
        # graph in the batch
        self.node_slices = []
        self.edge_slices = []
        tot_n = 0
        tot_e = 0
        for n, e in zip(self.dgl().batch_num_nodes(), self.dgl().batch_num_edges()):
            n = int(n)
            e = int(e)
            self.node_slices.append(slice(tot_n,tot_n+n))
            self.edge_slices.append(slice(tot_e,tot_e+e))
            tot_n += n
            tot_e += e

    def __len__(self) -> int:
        return self.dgl().batch_size

    def __getitem__(self, index: int) -> Graph[N, E]:
        if isinstance(index, int):
            if index >= len(self):
                raise IndexError()
        else:
            raise NotImplementedError()
        return GraphBatchView(self, index)

    def item_type(self):
        return self._graph_type

class GraphBatchView(BatchViewBase[G]):

    _internal_attribs = [ "_batch", "_index" ]
    _batch: Batch[G]
    _index: Union[int, slice] # either int or slice
    _node_type_tree: TypeTree
    _edge_type_tree: TypeTree

    def __init__(self, batch: Batch, index: Union[int, slice]):
        self._batch = batch
        self._index = index
        self._node_type_tree = batch._node_type_tree
        self._edge_type_tree = batch._edge_type_tree

    @property
    def ndata(self) -> Batch[N]:
        idxs = self._batch.node_slices[self._index]
        return self._batch.ndata[idxs]

    @property
    def edata(self) -> Batch[E]:
        idxs = self._batch.edge_slices[self._index]
        return self._batch.edata[idxs]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        node_idxs = self._batch.node_slices[self._index]
        n0 = node_idxs.start
        node_idxs_expl = range(len(self._batch.ndata))[node_idxs]
        ret = []
        for n1, n2 in self._batch.edges:
            if n1 in node_idxs_expl:
                assert n2 in node_idxs_expl
                ret.append((n1 - n0, n2 - n0))
        return ret

    def dgl(self):
        return dgl.unbatch(self._batch.dgl())[self._index]

    def get_type(self):
        return self._batch._graph_type

    def __repr__(self):
        return GraphBase.__repr__(self)
        
    @staticmethod
    def get_batch_type():
        return GraphBatch
        