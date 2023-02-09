import torch
from src.terrace.dataframe import DFRow, merge
from src.terrace.batch import Batch, collate

def test_df_row():
    rows = []
    for i in range(3):
        rows.append(DFRow(a=i, b="test"))
        rows[-1]["c"] = 3.0

    batch = Batch(rows)
    assert batch.a.shape == (3,)
    assert isinstance(batch.b, list) and len(batch.b) == 3
    assert batch.c.shape == (3,)

    batch = collate([item for item in batch])
    assert batch.a.shape == (3,)
    assert isinstance(batch.b, list) and len(batch.b) == 3
    assert batch.c.shape == (3,)

def test_merge():
    rows = []
    for i in range(3):
        rows.append(DFRow(a=i))
        rows[-1] = merge((rows[-1], DFRow(b="test")))
    batch = Batch(rows)
    batch = merge((batch, Batch(DFRow, c=torch.zeros(3,))))

    assert batch.a.shape == (3,)
    assert isinstance(batch.b, list) and len(batch.b) == 3
    assert batch.c.shape == (3,)

    batch = collate([item for item in batch])
    assert batch.a.shape == (3,)
    assert isinstance(batch.b, list) and len(batch.b) == 3
    assert batch.c.shape == (3,)
