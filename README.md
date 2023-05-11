# Terrace

[Documentation](https://terrace.readthedocs.io) | [Code](https://github.com/mixarcid/terrace) | [PyPI](https://pypi.org/project/terrace/)

Welcome to Terrace, a collection of high-level utilities for writing concise and maintainable PyTorch code. I've been using PyTorch in my own work for a while now, and I developed these tools to boost my productivity. I'm now ready to share them with the world -- I hope you find Terrace to be as helpful as I have.

Terrace provides two major features: Modules and Batches. Terrace Modules allow you to more concisely define your PyTorch models entirely in the `forward` method. Batches allow you to place all your data in nice classes rather than swinging around a bunch of raw tensors like a barbarian.

## Modules

If you're writing a vanilla PyTorch model, you need to populate both the `__init__` and `forward` methods of your model. `__init__` specifies all the submodules of your model, including all the input and output tensor shapes. `forward` specifies how to use all these submodules in order to run your computation. A simple neural network with a single hidden layer might look something like this:


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_nn = nn.Linear(128, 64)
        self.out_nn = nn.Linear(64, 1)

    def forward(self, x):
        hid = F.relu(self.hidden_nn(x))
        return self.out_nn(hid)
    
model = SimpleNetwork()
x = torch.randn(16, 128)
out = model(x)
print(out.shape)
```

    torch.Size([16, 1])


There are two annoyances with this approach. First, the overall structure of the network is repeated in both the `__init__` and `forward` methods. This may be fine for a toy example, but it gets pretty annoying with larger networks. Every time you make a change to the architecture, you need to constantly scroll between `__init__` and `forward`.

Additionally, you need to specify beforehand what the input shapes to the model will be. In this example, changing the input dimension from 128 to 256 would require changing code in two places. For models with many inputs with shapes, this can be a real pain.

Terrace `Modules` solve both these problems by allowing you to code the entire model in the `forward` method. Here's how the same model would be written with Terrace.


```python
import terrace as ter

class BetterSimpleNetwork(ter.Module):

    def forward(self, x):
        self.start_forward()
        # with the LazyLinear layers, we only need to specify the output dimension
        hid = F.relu(self.make(ter.LazyLinear, 64)(x))
        out = self.make(ter.LazyLinear, 1)(hid)
        return out
    
x = torch.randn(16, 128)
model = BetterSimpleNetwork()
out = model(x)
print(out.shape)
```

    torch.Size([16, 1])


The first time this model is run, the `make` calls with create new linear layers, each of which lazily creates their weight matrices based on their inputs. Writing complex models is now easier, faster, and just more fun.

There are some important caveats to this approach, so please make sure to check out the [documentation](https://terrace.readthedocs.io/en/latest/guides/module.html).

## Batches

Out of the box, PyTorch can collate tensors (and tuples of tensors) into batches. But what about arbitrary classes? If your neural network is dealing with complex datatypes, structuring your data in classes is the solution. By inheriting from Terrace's `Batchable` class, you can create datasets of arbitrary objects.

Let's say you're working with people who have faces and names


```python
MAX_NAME_LEN = 128
IMG_SIZE = 256
class Person(ter.Batchable):
    
    face: torch.Tensor
    name: torch.Tensor
    
    def __init__(self): # dummy person data
        self.face = torch.zeros((3, IMG_SIZE, IMG_SIZE))
        self.name = torch.zeros((MAX_NAME_LEN,))
```

Now you can collate multiple people them into batches of people. These batches behave like people whose member data are batchified.


```python
dave = Person()
rhonda = Person()
batch = ter.collate([dave, rhonda])
print(batch)
```

    Batch[Person](
       face=Tensor(shape=torch.Size([2, 3, 256, 256]), dtype=torch.float32)
       name=Tensor(shape=torch.Size([2, 128]), dtype=torch.float32)
    )


With Terrace's ``Dataloader`` class, you can make your datasets return people directly.


```python
class PersonDataset(torch.utils.data.Dataset):
    
    def __len__(self):
        return 16
    
    def __getitem__(self, index):
        return Person()
    
batch_size = 8
dataset = PersonDataset()
loader = ter.DataLoader(dataset, batch_size=batch_size)
for batch in loader:
    print(batch)
```

    Batch[Person](
       face=Tensor(shape=torch.Size([8, 3, 256, 256]), dtype=torch.float32)
       name=Tensor(shape=torch.Size([8, 128]), dtype=torch.float32)
    )
    Batch[Person](
       face=Tensor(shape=torch.Size([8, 3, 256, 256]), dtype=torch.float32)
       name=Tensor(shape=torch.Size([8, 128]), dtype=torch.float32)
    )


Terrace also has a higher-level interface for graph data, and several more features. Check out the [documentation](https://terrace.readthedocs.io)  for more info!

## Getting started

If you're interested in using Terrace for your own work, simply install via pip.

```bash
pip install terrace
```

If you find Terrace useful, or find a bug, or have an idea for further functionality, please reach out! [Email me](mailto:mixarcidiacono@gmail.com) or find me on [Twitter](https://twitter.com/mixarcid). 

## Disclaimer
At the moment, Terrace is very much a work in progress. The API is subject to change and there are likely many bugs. In its current state, I would not recommend this package for production use.
